"""
scripts/generate.py

Phase 4 — Inference.
Generates a taiko .osz file from an audio file using a trained checkpoint.

Usage:
    python scripts/generate.py \
        --audio "path/to/song.mp3" \
        --output output/ \
        --checkpoint checkpoints/best.pt \
        --difficulty 5.0 \
        --style standard
"""

from __future__ import annotations
import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from taiko.data.audio import MelExtractor
from taiko.data.tokenizer import TaikoVocabulary, OsuTaikoSerializer
from taiko.data.osu_parser import TaikoBeatmap, TaikoNote, TimingPoint
from taiko.data.beat_snap import detect_bpm, snap_note_events, build_timing_points_from_bpm
from taiko.model.model import TaikoMapper, TaikoModelConfig


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

STYLE_PRESETS = {
    "standard": {"nps": 6.0,  "don_ratio": 0.50, "big_ratio": 0.05},
    "stream":   {"nps": 12.0, "don_ratio": 0.55, "big_ratio": 0.03},
    "speed":    {"nps": 10.0, "don_ratio": 0.50, "big_ratio": 0.04},
    "tech":     {"nps": 7.0,  "don_ratio": 0.45, "big_ratio": 0.08},
}


def build_conditioning(vocab, difficulty, style, device):
    preset   = STYLE_PRESETS.get(style, STYLE_PRESETS["standard"])
    cond_ids = vocab.conditioning_ids(
        star_rating=difficulty,
        notes_per_second=preset["nps"],
        don_ratio=preset["don_ratio"],
        big_ratio=preset["big_ratio"],
        overall_difficulty=min(10.0, difficulty),
    )
    return torch.tensor(cond_ids, dtype=torch.long, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Segment generation
# ---------------------------------------------------------------------------

def generate_full_map(model, mel, cond_ids, vocab, device,
                      window_ms=8192, overlap_ms=1000,
                      bpm_beat_length=500.0, bpm_offset=0.0,
                      temperature=0.95, top_p=0.92, cfg_scale=1.5):
    from taiko.data.audio import SAMPLE_RATE, HOP_LENGTH

    ms_per_frame   = HOP_LENGTH / SAMPLE_RATE * 1000.0
    total_frames   = mel.shape[1]
    total_ms       = total_frames * ms_per_frame
    window_frames  = int(window_ms  / ms_per_frame)
    step_ms        = window_ms - overlap_ms
    step_frames    = int(step_ms / ms_per_frame)
    n_windows      = max(1, int((total_ms - overlap_ms) / step_ms) + 1)

    print(f"Song:    {total_ms/1000:.1f}s | Windows: {n_windows} × {window_ms/1000:.0f}s")

    NOTE_TOKENS = {
        "HIT_DON", "HIT_KAT", "BIG_DON", "BIG_KAT",
        "ROLL_START", "ROLL_END", "DENDEN_START", "DENDEN_END"
    }

    all_notes = []
    model.eval()

    for i in range(n_windows):
        start_ms    = i * step_ms
        end_ms      = start_ms + window_ms
        start_frame = int(start_ms / ms_per_frame)
        end_frame   = min(start_frame + window_frames, total_frames)

        mel_window = mel[:, start_frame:end_frame]
        if mel_window.shape[1] < window_frames:
            pad = torch.zeros(mel.shape[0], window_frames - mel_window.shape[1])
            mel_window = torch.cat([mel_window, pad], dim=1)

        mel_window = mel_window.unsqueeze(0).to(device)

        print(f"  [{i+1}/{n_windows}] {start_ms/1000:.1f}s-{end_ms/1000:.1f}s ...", end=" ", flush=True)
        t0 = time.time()

        tokens = model.generate(
            mel_window, cond_ids,
            max_new_tokens=512,
            temperature=temperature,
            top_p=top_p,
            cfg_scale=cfg_scale,
        )
        print(f"{len(tokens)} tokens ({time.time()-t0:.1f}s)")

        # Decode beat-relative tokens to (abs_time_ms, token_str)
        from taiko.data.tokenizer import steps_to_ms, ms_to_steps
        # abs_steps starts at window start so times decode to absolute song positions
        abs_steps = ms_to_steps(start_ms, bpm_beat_length, bpm_offset)
        keep_from = start_ms + (overlap_ms / 2 if i > 0 else 0)
        keep_to   = end_ms   - (overlap_ms / 2 if i < n_windows - 1 else 0)

        j = 0
        while j < len(tokens):
            tid = tokens[j]
            if tid in (vocab.SOS_ID, vocab.PAD_ID):
                j += 1; continue
            if tid == vocab.EOS_ID:
                break
            if tid == vocab.SILENCE_ID:
                j += 1; continue
            if vocab.is_beat_token(tid):
                abs_steps += vocab.beat_token_to_steps(tid)
                abs_time_ms = steps_to_ms(abs_steps, bpm_beat_length, bpm_offset)
                if j + 1 < len(tokens):
                    note_tok = vocab.decode(tokens[j + 1])
                    if keep_from <= abs_time_ms < keep_to and note_tok in NOTE_TOKENS:
                        all_notes.append((int(abs_time_ms), note_tok))
                j += 2; continue
            j += 1

    # Sort + deduplicate within 10ms
    all_notes.sort(key=lambda x: x[0])
    deduped, last_t = [], -999
    for t, tok in all_notes:
        if t - last_t >= 10:
            deduped.append((t, tok)); last_t = t

    print(f"Raw notes: {len(deduped)}")
    return deduped


# ---------------------------------------------------------------------------
# Notes → TaikoBeatmap
# ---------------------------------------------------------------------------

def notes_to_beatmap(note_events, audio_path, bpm, offset_ms,
                     difficulty, style, title, artist):
    bm = TaikoBeatmap()
    bm.title          = title
    bm.artist         = artist
    bm.creator        = "TaikoAI"
    bm.version        = f"AI {style.capitalize()} {difficulty:.1f}*"
    bm.audio_filename = audio_path.name
    bm.overall_difficulty = min(10.0, difficulty)
    bm.hp_drain           = min(10.0, difficulty * 0.8)
    bm.slider_multiplier  = 1.4
    bm.slider_tick_rate   = 1.0

    beat_length = 60_000.0 / bpm
    bm.timing_points = [TimingPoint(
        time=int(offset_ms),
        beat_length=beat_length,
        meter=4,
        uninherited=True,
    )]

    note_map = {
        "HIT_DON": "don", "HIT_KAT": "kat",
        "BIG_DON": "big_don", "BIG_KAT": "big_kat",
        "ROLL_START": "roll", "DENDEN_START": "denden",
    }

    notes = []
    for abs_time, tok in note_events:
        if tok in note_map:
            note_type = note_map[tok]
            end_time  = abs_time + int(beat_length * 2) if note_type in ("roll", "denden") else 0
            notes.append(TaikoNote(time=abs_time, note_type=note_type, end_time=end_time))
        elif tok == "ROLL_END":
            for n in reversed(notes):
                if n.note_type == "roll":
                    n.end_time = max(abs_time, n.time + int(beat_length))
                    break
        elif tok == "DENDEN_END":
            for n in reversed(notes):
                if n.note_type == "denden":
                    n.end_time = max(abs_time, n.time + int(beat_length * 2))
                    break

    bm.notes = notes
    bm.compute_stats()
    return bm


# ---------------------------------------------------------------------------
# Write .osz
# ---------------------------------------------------------------------------

def write_osz(bm: TaikoBeatmap, audio_path: Path, output_path: Path):
    """Package .osu + audio into a .osz file."""
    from taiko.data.tokenizer import OsuTaikoSerializer
    osu_text = OsuTaikoSerializer().serialize(bm, audio_path.name)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            f"{bm.title} [{bm.version}].osu",
            osu_text.encode("utf-8")
        )
        z.write(audio_path, audio_path.name)

    print(f"Saved .osz: {output_path}")
    print(f"  → Double-click to import into osu!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",       required=True)
    parser.add_argument("--output",      default="output/")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    parser.add_argument("--difficulty",  type=float, default=5.0)
    parser.add_argument("--style",       default="standard",
                        choices=["standard", "stream", "speed", "tech"])
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p",       type=float, default=0.92)
    parser.add_argument("--cfg-scale",   type=float, default=1.5)
    parser.add_argument("--title",       default="")
    parser.add_argument("--artist",      default="")
    parser.add_argument("--snap-tolerance", type=float, default=20.0,
                        help="Max ms distance to snap note to beat grid")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load checkpoint ----------------------------------------------- #
    print(f"Loading: {args.checkpoint}")
    ckpt   = torch.load(args.checkpoint, map_location=device)
    vocab  = TaikoVocabulary()
    cfg    = ckpt["config"]
    m_cfg  = cfg["model"]

    model_config = TaikoModelConfig(
        vocab_size=len(vocab),
        n_mels=m_cfg["n_mels"],
        encoder_d_model=m_cfg["encoder_d_model"],
        d_model=m_cfg["d_model"],
        nhead=m_cfg["nhead"],
        num_layers=m_cfg["num_layers"],
        dim_feedforward=m_cfg["dim_feedforward"],
        dropout=0.0,
        max_seq_len=m_cfg["max_seq_len"],
        n_cond_tokens=m_cfg["n_cond_tokens"],
    )
    model = TaikoMapper(model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model ready (step {ckpt['step']}, val_loss {ckpt['best_val_loss']:.4f})")

    # ---- Extract mel ---------------------------------------------------- #
    audio_path = Path(args.audio)
    print(f"Extracting mel...")
    mel = torch.from_numpy(MelExtractor().extract(audio_path))

    # ---- Detect BPM ----------------------------------------------------- #
    print(f"Detecting BPM...")
    bpm, beat_times = detect_bpm(audio_path)
    offset_ms = float(beat_times[0]) * 1000.0 if len(beat_times) > 0 else 0.0
    print(f"BPM: {bpm:.1f} | First beat: {offset_ms:.0f}ms")

    # ---- Conditioning --------------------------------------------------- #
    cond_ids = build_conditioning(vocab, args.difficulty, args.style, device)
    print(f"Difficulty: {args.difficulty}★  Style: {args.style}")

    # ---- Generate ------------------------------------------------------- #
    bpm_beat_length = 60_000.0 / bpm
    note_events = generate_full_map(
        model, mel, cond_ids, vocab, device,
        bpm_beat_length=bpm_beat_length,
        bpm_offset=offset_ms,
        temperature=args.temperature,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
    )

    if not note_events:
        print("WARNING: No notes generated.")
        return

    # ---- Snap to BPM grid ----------------------------------------------- #
    print(f"Snapping to BPM grid (tolerance={args.snap_tolerance}ms)...")
    note_events = snap_note_events(
        note_events,
        bpm=bpm,
        offset_ms=offset_ms,
        tolerance_ms=args.snap_tolerance,
        discard_unsnapped=False,
    )
    print(f"After snap: {len(note_events)} notes")

    # ---- Build beatmap -------------------------------------------------- #
    title  = args.title  or audio_path.stem
    artist = args.artist or "Unknown Artist"
    bm = notes_to_beatmap(note_events, audio_path, bpm, offset_ms,
                          args.difficulty, args.style, title, artist)
    print(f"Stats: {bm.note_count} notes | {bm.notes_per_second:.1f} nps | don {bm.don_ratio:.0%}")

    # ---- Write .osz ----------------------------------------------------- #
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in title if c.isalnum() or c in " -_")[:40].strip()
    osz_path = output_dir / f"{safe} [AI {args.style} {args.difficulty:.1f}].osz"
    write_osz(bm, audio_path, osz_path)


if __name__ == "__main__":
    main()