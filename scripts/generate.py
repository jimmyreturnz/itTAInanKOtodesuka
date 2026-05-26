"""
scripts/generate.py

Generate a taiko beatmap from an audio file using trained models.

Usage:
    python scripts/generate.py --audio path/to/song.mp3
    python scripts/generate.py --audio path/to/song.mp3 --difficulty 5.5 --style stream
    python scripts/generate.py --audio path/to/song.mp3 --cfg-scale 2.0 --steps 100
    python scripts/generate.py --audio path/to/song.mp3 --no-refine   # skip grid snap
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
from taiko.data.tensor_repr import tensor_to_beatmap, FRAME_MS
from taiko.data.timing_refine import apply_timing_refinement
from taiko.data.tokenizer import OsuTaikoSerializer
from taiko.model.diffusion import TaikoDiffusion
from taiko.model.model_config import PROFILES, get_profile


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AE_CKPT        = "checkpoints/autoencoder/best.pt"
DIFF_CKPT      = "checkpoints/diffusion/best.pt"
OUTPUT_DIR     = Path("outputs")

STYLE_MAP      = {"standard": 0, "stream": 1, "speed": 2, "tech": 3}
COMPRESSION    = 16   # autoencoder compression ratio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(
    audio_path:  str,
    ae_ckpt:     str = AE_CKPT,
    diff_ckpt:   str = DIFF_CKPT,
    difficulty:  float = 5.0,
    style:       str = "standard",
    cfg_scale:   float = 1.5,
    ddim_steps:  int = 50,
    output_dir:  Path = OUTPUT_DIR,
    refine_timing: bool = True,
    profile:       str = "p1",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return

    if not Path(ae_ckpt).exists():
        print(f"ERROR: Autoencoder checkpoint not found: {ae_ckpt}")
        return

    if not Path(diff_ckpt).exists():
        print(f"ERROR: Diffusion checkpoint not found: {diff_ckpt}")
        return

    style_int = STYLE_MAP.get(style, 0)

    # ---- Load model -------------------------------------------------------- #
    ckpt = torch.load(diff_ckpt, map_location="cpu")
    profile_name = ckpt.get("profile", profile)
    prof = get_profile(profile_name)
    print(f"Loading model (profile={profile_name})...")

    model = TaikoDiffusion(
        autoencoder_ckpt    = ae_ckpt,
        timesteps           = 1000,
        beta_schedule       = "linear",
        z_channels          = 16,
        n_mels              = 128,
        audio_base_channels = prof.audio_base_channels,
        audio_channel_mult  = list(prof.audio_channel_mult),
        unet_base_channels  = prof.unet_base_channels,
        unet_channel_mult   = list(prof.unet_channel_mult),
        unet_num_res_blocks = prof.unet_num_res_blocks,
        n_styles            = 4,
        cfg_dropout         = prof.cfg_dropout,
        use_checkpoint      = False,
        use_s4              = prof.use_s4,
    ).to(device)

    model.unet_model.load_state_dict(ckpt["unet"], strict=False)
    model.wave_model.load_state_dict(ckpt["wave"], strict=False)
    model.eval()
    print(f"Loaded diffusion checkpoint: step={ckpt['step']}, val_loss={ckpt['best_val']:.4f}")

    # ---- Extract mel ------------------------------------------------------- #
    print(f"Extracting mel from {audio_path.name}...")
    extractor = MelExtractor()
    mel = extractor.extract(str(audio_path))        # [128, T_mel]
    mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(device)  # [1, 128, T_mel]
    print(f"Mel shape: {tuple(mel_tensor.shape)}")

    # ---- Audio BPM hint (used after decode for timing refinement) ---------- #
    print("Detecting BPM (audio hint)...")
    bpm, beat_times = detect_bpm(audio_path)
    audio_offset_ms = float(beat_times[0] * 1000) if len(beat_times) > 0 else 0.0
    print(f"Audio BPM: {bpm:.1f}, first beat offset: {audio_offset_ms:.0f} ms")

    # ---- Compute latent length --------------------------------------------- #
    # mel frames == beatmap frames at 20 ms hop (do not divide by 2)
    mel_frames      = mel_tensor.shape[2]
    beatmap_frames  = mel_frames          # convert mel frames to beatmap frames
    latent_length   = beatmap_frames // COMPRESSION
    duration_sec    = beatmap_frames * FRAME_MS / 1000
    print(f"Song duration  : {duration_sec:.1f}s")
    print(f"Beatmap frames : {beatmap_frames}")
    print(f"Latent length  : {latent_length}")

    # ---- Generate ---------------------------------------------------------- #
    print(f"\nGenerating beatmap...")
    print(f"  Difficulty : {difficulty}")
    print(f"  Style      : {style} ({style_int})")
    print(f"  CFG scale  : {cfg_scale}")
    print(f"  DDIM steps : {ddim_steps}")

    with torch.no_grad():
        z = model.generate(
            mel           = mel_tensor,
            difficulty    = difficulty,
            style         = style_int,
            latent_length = latent_length,
            ddim_steps    = ddim_steps,
            cfg_scale     = cfg_scale,
            device        = device,
        )
        beatmap_tensor = model.decode(z)[0].cpu().numpy()   # [7, T]

    print(f"Generated tensor shape: {beatmap_tensor.shape}")
    print(f"Channel sums: {beatmap_tensor.sum(axis=1).round(2).tolist()}")

    # ---- Convert to beatmap ------------------------------------------------ #
    print("\nConverting to beatmap...")
    bm = tensor_to_beatmap(
        beatmap_tensor,
        bpm                = bpm,
        offset_ms          = audio_offset_ms,
        title              = audio_path.stem,
        artist             = "",
        version            = f"AI {style.capitalize()} {difficulty:.1f}*",
        audio_filename     = audio_path.name,
        overall_difficulty = min(10.0, difficulty),
    )

    if refine_timing and bm.note_count > 0:
        print("\nRefining timing (Mug-style fit + grid snap)...")
        apply_timing_refinement(
            bm,
            audio_path=audio_path,
            audio_bpm=bpm,
            audio_offset_ms=audio_offset_ms,
            verbose=True,
        )

    print(f"Notes     : {bm.note_count}")
    print(f"NPS       : {bm.notes_per_second:.1f}")
    print(f"Don ratio : {bm.don_ratio:.0%}")
    print(f"Duration  : {bm.duration_ms/1000:.1f}s")

    if bm.note_count == 0:
        print("\nWARNING: Generated map has 0 notes.")
        print("The model may need more training, or try a lower CFG scale.")

    # ---- Export as .osz ---------------------------------------------------- #
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(c for c in audio_path.stem if c.isalnum() or c in " -_")[:40]
    osu_name   = f"{safe_title} [AI {style.capitalize()} {difficulty:.1f}].osu"
    osz_path   = output_dir / f"{safe_title} [AI {style.capitalize()} {difficulty:.1f}].osz"

    osu_text = OsuTaikoSerializer().serialize(bm, audio_path.name)

    with zipfile.ZipFile(osz_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(osu_name, osu_text.encode("utf-8"))
        z.write(str(audio_path), audio_path.name)

    print(f"\nSaved: {osz_path}")
    print("Double-click the .osz to import into osu!")
    return osz_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a taiko beatmap from audio")
    ap.add_argument("--audio",      required=True,        help="Path to audio file (.mp3/.ogg/.wav)")
    ap.add_argument("--ae-ckpt",    default=AE_CKPT,      help="Autoencoder checkpoint path")
    ap.add_argument("--diff-ckpt",  default=DIFF_CKPT,    help="Diffusion checkpoint path")
    ap.add_argument("--difficulty", type=float, default=5.0, help="Target star rating (default: 5.0)")
    ap.add_argument("--style",      default="standard",
                    choices=["standard", "stream", "speed", "tech"],
                    help="Map style (default: standard)")
    ap.add_argument("--cfg-scale",  type=float, default=1.5, help="CFG scale (default: 1.5)")
    ap.add_argument("--steps",      type=int,   default=50,  help="DDIM steps (default: 50)")
    ap.add_argument("--output-dir", default="outputs",    help="Output directory (default: outputs/)")
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip post-gen BPM fit + grid snap (Mug gridify)")
    ap.add_argument("--profile", default="p1", choices=list(PROFILES.keys()),
                    help="Model profile if checkpoint has no profile field")
    args = ap.parse_args()

    generate(
        audio_path      = args.audio,
        ae_ckpt         = args.ae_ckpt,
        diff_ckpt       = args.diff_ckpt,
        difficulty      = args.difficulty,
        style           = args.style,
        cfg_scale       = args.cfg_scale,
        ddim_steps      = args.steps,
        output_dir      = Path(args.output_dir),
        refine_timing   = not args.no_refine,
        profile         = args.profile,
    )