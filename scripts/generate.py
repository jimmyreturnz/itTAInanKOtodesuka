"""
scripts/generate.py

Generate a playable .osz from an audio file.

    python scripts/generate.py --audio song.mp3 --difficulty 5.5
    python scripts/generate.py --audio song.mp3 --difficulty 7 --style stream
    python scripts/generate.py --audio song.mp3 --preset tech --bpm 180 --offset 317
    python scripts/generate.py --audio song.mp3 --reference "some map.osu"

Conditioning actually reaches the model here. The previous version passed
neither NPS nor motif and used constructor arguments the model no longer
accepted, so it raised on import -- and had it run, every generation would have
been unconditional regardless of the flags, because unsupplied conditioning is
the null embedding.

Tempo is an input, not something the model invents. Give --bpm and --offset when
you know them; otherwise they are detected from the audio and printed so you can
correct them. Getting the grid right is most of getting the chart right.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from taiko.data.audio import MelExtractor
from taiko.data.beat_snap import detect_bpm
from taiko.data.conditioning import (
    STYLE_NULL, normalise_avg_nps, normalise_difficulty, normalise_peak_nps,
    style_to_int,
)
from taiko.data.frames import FRAME_MS, describe, frames_to_sec
from taiko.data.motif import (
    MOTIF_NAMES, PRESETS, beat_frames_from_bpm, compute_motif, describe_motif,
    get_preset,
)
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.tensor_repr import (
    beatmap_to_tensors, tensor_to_beatmap, timing_stream_from_bpm,
)
from taiko.data.timing_refine import apply_timing_refinement
from taiko.data.osu_writer import OsuTaikoSerializer
from taiko.model.diffusion import TaikoDiffusion
from taiko.model.sampling import generate_song


def load_model(diffusion_ckpt: Path, ae_ckpt: Path, device: torch.device):
    ckpt = torch.load(diffusion_ckpt, map_location="cpu", weights_only=False)
    profile = ckpt.get("profile", "p1")

    model = TaikoDiffusion(
        autoencoder_ckpt=str(ae_ckpt),
        profile=profile,
        prediction_type=ckpt.get("prediction_type", "v"),
    )
    model.unet_model.load_state_dict(ckpt["unet"])
    model.wave_model.load_state_dict(ckpt["wave"])

    # Sample from the EMA weights. They are what validation measured and what
    # the model is actually good at; the live weights are wherever the last
    # gradient step happened to leave them.
    ema_state = ckpt.get("ema")
    if ema_state:
        shadow = ema_state["shadow"]
        with torch.no_grad():
            for param, value in zip(model.trainable_parameters(), shadow):
                param.data.copy_(value.to(param.dtype))
        print(f"Using EMA weights ({ema_state['step']} updates)")
    else:
        print("WARNING: checkpoint has no EMA weights; sample quality will suffer")

    ae_ckpt_data = torch.load(ae_ckpt, map_location="cpu", weights_only=False)
    threshold = ae_ckpt_data.get("onset_threshold", 0.5)

    model = model.to(device).eval()
    print(f"Model: profile {profile}, step {ckpt.get('step', '?')}, "
          f"val {ckpt.get('best_val', float('nan')):.5f}, onset threshold {threshold}")
    return model, threshold, ckpt


def resolve_motif(args, parser: OsuTaikoParser) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Turn --preset / --reference / --motif into a vector and its mask."""
    if args.reference:
        reference = Path(args.reference)
        if not reference.exists():
            raise FileNotFoundError(f"reference map not found: {reference}")
        bm = parser.parse_file(reference)
        chart, _ = beatmap_to_tensors(bm)
        bpm = 0.0
        for tp in bm.timing_points:
            if tp.uninherited and tp.beat_length > 0:
                bpm = 60_000.0 / tp.beat_length
                break
        motif = compute_motif(chart, beat_frames_from_bpm(bpm))
        print(f"Style extracted from {reference.name}:")
        print(describe_motif(motif))
        return motif, np.ones_like(motif)

    if args.preset:
        motif = get_preset(args.preset)
        print(f"Style preset {args.preset!r}:")
        print(describe_motif(motif))
        return motif, np.ones_like(motif)

    if args.motif:
        motif = np.asarray(args.motif, dtype=np.float32)
        if motif.size != len(MOTIF_NAMES):
            raise ValueError(f"--motif needs {len(MOTIF_NAMES)} values, got {motif.size}")
        return motif, np.ones_like(motif)

    # Nothing requested. An all-zero mask means "unspecified", which is not the
    # same as asking for a chart with zero of everything.
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate an osu!taiko map from audio")
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--diffusion", type=Path, default=Path("checkpoints/diffusion/best.pt"))
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))

    ap.add_argument("--difficulty", type=float, default=5.0, help="target star rating")
    ap.add_argument("--style", default=None,
                    choices=["standard", "stream", "speed", "tech"])
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS),
                    help="named motif preset")
    ap.add_argument("--reference", default=None,
                    help="an .osu file to copy the style of")
    ap.add_argument("--motif", type=float, nargs="+", default=None,
                    help="16 raw motif values (advanced)")
    ap.add_argument("--avg-nps", type=float, default=None)
    ap.add_argument("--peak-nps", type=float, default=None)

    ap.add_argument("--bpm", type=float, default=None, help="skip tempo detection")
    ap.add_argument("--offset", type=float, default=None, help="first beat, ms")
    ap.add_argument("--meter", type=int, default=4)

    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--window-frames", type=int, default=None,
                    help="default: whatever the checkpoint trained with")
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the checkpoint's onset threshold")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip the post-generation grid snap")
    args = ap.parse_args()

    print(describe())

    for path, what in ((args.audio, "audio"), (args.diffusion, "diffusion checkpoint"),
                       (args.ae, "autoencoder checkpoint")):
        if not Path(path).exists():
            print(f"ERROR: {what} not found: {path}")
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_threshold, ckpt = load_model(args.diffusion, args.ae, device)
    threshold = args.threshold if args.threshold is not None else ckpt_threshold

    window = args.window_frames or ckpt.get("window_frames", 1536)
    overlap = args.overlap if args.overlap is not None else window // 2

    # ---- audio -------------------------------------------------------- #
    print(f"\nExtracting mel from {args.audio.name} ...")
    mel = MelExtractor().extract(args.audio)
    total_frames = mel.shape[1]
    print(f"  {total_frames} frames = {frames_to_sec(total_frames):.1f}s")

    # ---- tempo -------------------------------------------------------- #
    if args.bpm is not None:
        bpm = args.bpm
        offset = args.offset if args.offset is not None else 0.0
        print(f"Tempo: {bpm:.1f} BPM, offset {offset:.0f} ms  (supplied)")
    else:
        print("Detecting tempo ...")
        bpm, beats = detect_bpm(args.audio)
        offset = float(beats[0] * 1000) if len(beats) else 0.0
        if args.offset is not None:
            offset = args.offset
        print(f"Tempo: {bpm:.1f} BPM, offset {offset:.0f} ms  (detected)")
        print("  If the chart feels off-grid, re-run with the real values:")
        print(f"    --bpm <bpm> --offset <ms>")

    timing = timing_stream_from_bpm(bpm, offset, total_frames, meter=args.meter)

    # ---- conditioning -------------------------------------------------- #
    parser = OsuTaikoParser()
    motif, motif_mask = resolve_motif(args, parser)
    style = style_to_int(args.style) if args.style else STYLE_NULL

    print(f"\nGenerating:")
    print(f"  difficulty  {args.difficulty}*")
    print(f"  style       {args.style or 'unspecified'}")
    print(f"  guidance    {args.cfg_scale}   steps {args.steps}")
    print(f"  window      {window} frames, overlap {overlap}")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    chart = generate_song(
        model,
        mel=torch.from_numpy(mel).unsqueeze(0),
        timing=torch.from_numpy(timing).unsqueeze(0),
        difficulty=normalise_difficulty(args.difficulty),
        style=style,
        avg_nps=normalise_avg_nps(args.avg_nps) if args.avg_nps else None,
        peak_nps=normalise_peak_nps(args.peak_nps) if args.peak_nps else None,
        motif=motif,
        motif_mask=motif_mask,
        window_frames=window,
        overlap_frames=overlap,
        ddim_steps=args.steps,
        cfg_scale=args.cfg_scale,
        eta=args.eta,
        generator=generator,
    )[0].cpu().numpy()

    # ---- decode -------------------------------------------------------- #
    style_label = args.preset or args.style or "AI"
    bm = tensor_to_beatmap(
        chart, bpm=bpm, offset_ms=offset, threshold=threshold, meter=args.meter,
        title=args.audio.stem, artist="",
        version=f"{style_label.capitalize()} {args.difficulty:.1f}",
        audio_filename=args.audio.name,
        overall_difficulty=min(10.0, args.difficulty),
    )

    if not args.no_refine and bm.note_count > 0:
        print("\nSnapping to the beat grid ...")
        apply_timing_refinement(
            bm, audio_path=args.audio, audio_bpm=bpm,
            audio_offset_ms=offset, meter=args.meter,
            trust_given_tempo=True, verbose=True,
        )

    print(f"\nResult:")
    print(f"  notes     {bm.note_count}")
    print(f"  nps       {bm.notes_per_second:.2f}")
    print(f"  don/kat   {bm.don_ratio:.0%} / {1 - bm.don_ratio:.0%}")
    print(f"  big       {bm.big_ratio:.1%}")
    print(f"  rolls     {bm.roll_count}   dendens {bm.denden_count}")
    print(f"  duration  {bm.duration_ms / 1000:.1f}s")

    if bm.note_count == 0:
        print("\nThe map is empty. Try a lower --threshold or a lower --cfg-scale;")
        print("an undertrained model also produces this.")
        return 1

    # ---- package -------------------------------------------------------- #
    args.out.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in args.audio.stem if c.isalnum() or c in " -_")[:40].strip()
    name = f"{safe} [{style_label.capitalize()} {args.difficulty:.1f}]"
    osz_path = args.out / f"{name}.osz"

    with zipfile.ZipFile(osz_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}.osu", OsuTaikoSerializer().serialize(bm, args.audio.name))
        archive.write(str(args.audio), args.audio.name)

    print(f"\nSaved {osz_path}")
    print("Open it with osu! to import.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
