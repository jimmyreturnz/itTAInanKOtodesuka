"""
scripts/measure_section_density.py

Does the model put its hard sections where the mapper did? Round 6 (Jimmy):
"a part that is supposed to be hard is mapped easy, a part that is supposed
to be easy is mapped hard", and dense charts over silence.

Each held-out map is cut into 4-bar sections on its own red lines. Per
section: the ranked chart's NPS, the model's NPS, and the audio's loudness and
flux. Three ways to give the model per-window density (`window_nps`), same
seed each:

    unknown   None, as evaluate and blind_ab run today
    oracle    the ranked chart's own onsets per window, the training target:
              the ceiling, and whether the model obeys the input at all
    auto      generate.py --window-density auto (audio flux heuristic)

    python scripts/measure_section_density.py --per-band 10
    python scripts/measure_section_density.py --per-band 2 --modes unknown   # quick

Headline: Spearman of model section NPS against ranked section NPS. "swap" is
the share of sections in the ranked chart's top third that the model put in
its bottom third, or the other way round.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from scipy.stats import spearmanr

from evaluate import SHIPPED, breaks, break_note_rate, decode_variants, dominant_tempo, load_model
from taiko.data.audio_activity import activity, silence_rate, window_densities
from taiko.data.conditioning import (SR_BANDS, divisors_used, normalise_avg_nps,
                                     normalise_difficulty, normalise_peak_nps)
from taiko.data.decode import decode_on_grid
from taiko.data.frames import FRAME_MS
from taiko.data.grid import Grid
from taiko.data.preprocessed_dataset import WINDOW_FRAMES_DEFAULT, split_indices
from taiko.data.profile import record_conditioning
from taiko.data.shards import ShardReader, decode_timing_points
from taiko.data.tensor_repr import ONSET_CHANNELS, build_timing_stream
from taiko.eval.criteria import level_of
from taiko.model.sampling import generate_song, plan_windows

MODES = ("unknown", "oracle", "auto")
BARS = 4


def sections(grid: Grid, end_ms: float) -> list[tuple[float, float]]:
    """4-bar spans from the first red line, each sized by the red line it starts under."""
    out, t = [], grid.sections[0].offset_ms
    while t < end_ms:
        sec = grid.section_at(t)
        step = BARS * sec.meter * sec.ms_per_beat
        if t + step > 0:      # a red line before 0 ms: clip the first bar, keep the grid
            out.append((max(t, 0.0), min(t + step, end_ms)))
        t += step
    return out


def section_nps(times: np.ndarray, spans) -> np.ndarray:
    return np.asarray([((times >= a) & (times < b)).sum() / ((b - a) / 1000.0) for a, b in spans])


def section_mean(x: np.ndarray, spans) -> np.ndarray:
    return np.asarray([x[int(a / FRAME_MS):max(int(b / FRAME_MS), int(a / FRAME_MS) + 1)].mean()
                       for a, b in spans])


def swap_rate(model: np.ndarray, ranked: np.ndarray) -> float:
    """Share of the ranked top/bottom-third sections the model put in the opposite third."""
    lo_r, hi_r = np.quantile(ranked, [1 / 3, 2 / 3])
    lo_m, hi_m = np.quantile(model, [1 / 3, 2 / 3])
    hard, easy = ranked > hi_r, ranked < lo_r
    n = hard.sum() + easy.sum()
    return float(((hard & (model < lo_m)).sum() + (easy & (model > hi_m)).sum()) / n) if n else np.nan


def rho(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def oracle_windows(chart: np.ndarray, windows) -> list[float]:
    """The training target: onsets in the window per second (preprocessed_dataset.py)."""
    onsets = (chart[list(ONSET_CHANNELS)] > 0.5).sum(axis=0)
    return [normalise_avg_nps(onsets[s:e].sum() / ((e - s) * FRAME_MS / 1000.0)) for s, e in windows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffusion", type=Path, default=Path("checkpoints/diffusion/best.pt"))
    ap.add_argument("--ae", type=Path, default=Path("checkpoints/autoencoder/best.pt"))
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--per-band", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=45_000)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=Path("outputs/section_density_cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/section_density.json"))
    ap.add_argument("--match", nargs="+", default=None,
                    help="held-out maps whose 'title [version]' contains one of these, "
                         "instead of the per-band pool (e.g. a blind round's songs)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, ckpt = load_model(args.diffusion, args.ae, device, True)
    window = ckpt.get("window_frames", WINDOW_FRAMES_DEFAULT)
    step = ckpt.get("step")
    print(f"{args.diffusion}: step {step}, window {window}, modes {args.modes}")

    reader = ShardReader(args.shards)
    _, val_idx = split_indices(reader, val_ratio=0.05)
    rng = np.random.default_rng(args.seed)
    chosen = []
    for lo, hi, label in SR_BANDS:
        band = [i for i in val_idx if lo <= float(reader.records[i].get("difficulty", 0.0)) < hi]
        if args.match:
            chosen += [(i, label) for i in band if any(
                m in f"{reader.records[i].get('title')} [{reader.records[i].get('version')}]"
                for m in args.match)]
        else:
            chosen += [(int(i), label) for i in rng.permutation(band)[:args.per_band]]

    rows = []
    for n, (idx, band) in enumerate(chosen, start=1):
        record = reader.records[idx]
        frames = min(reader.chart_length(idx), reader.mel_length(idx), args.max_frames)
        points = decode_timing_points(record["timing_points"])
        grid = Grid(points)
        bpm, offset, meter = dominant_tempo(points)
        mel_np = reader.mel_window(idx, 0, frames)
        mel = torch.from_numpy(mel_np).unsqueeze(0)
        timing = torch.from_numpy(build_timing_stream(points, frames, start_frame=0)).unsqueeze(0)
        chart = reader.chart_window(idx, 0, frames)
        act = activity(mel_np)
        sr = float(record.get("difficulty", 5.0))
        nps = float(record.get("avg_nps", 0.0))
        windows = plan_windows(frames, window, window // 2)

        ranked = decode_on_grid(chart, points, threshold=0.5)
        ranked_t = np.asarray([x.time for x in ranked.notes if not x.is_long], dtype=float)
        spans = sections(grid, frames * FRAME_MS)
        r_nps = section_nps(ranked_t, spans)
        loud, flux = section_mean(act.loudness, spans), section_mean(act.flux, spans)
        self_rho = rho(r_nps, r_nps)
        assert np.isnan(self_rho) or abs(self_rho - 1.0) < 1e-9, "harness self-check"
        row = {"map": f"{record.get('title', '?')} [{record.get('version', '?')}]", "band": band,
               "sections": len(spans),
               "ranked": {"rho_loud": rho(r_nps, loud), "rho_flux": rho(r_nps, flux),
                          "silence": silence_rate(ranked.notes, act)}}
        gaps = breaks(ranked.notes, grid)

        for mode in args.modes:
            cached = args.cache / mode / f"{step}_{idx}_{args.seed}_{args.steps}_{args.cfg_scale}_{frames}.npy"
            if cached.exists():
                probs = np.load(cached)
            else:
                wnps = {"unknown": None, "oracle": oracle_windows(chart, windows),
                        "auto": window_densities(nps, windows, act) if nps else None}[mode]
                probs = generate_song(
                    model, mel=mel, timing=timing, difficulty=normalise_difficulty(sr),
                    avg_nps=normalise_avg_nps(nps) if nps else None,
                    peak_nps=normalise_peak_nps(float(record.get("peak_nps", 0.0))) or None,
                    **record_conditioning(record, model.features),
                    window_nps=wnps, window_frames=window, overlap_frames=window // 2,
                    ddim_steps=args.steps, cfg_scale=args.cfg_scale, progress=False,
                    generator=torch.Generator(device=device).manual_seed(args.seed + idx),
                )[0].cpu().numpy()
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.save(cached, probs)
            counts = record.get("snap_counts")
            shipped = decode_variants(probs, points, grid, threshold, bpm, offset, meter,
                                      divisors_used(counts) if counts else None,
                                      level_of(record.get("version", ""), sr))[SHIPPED]
            m_t = np.asarray([x.time for x in shipped.notes if not x.is_long], dtype=float)
            m_nps = section_nps(m_t, spans)
            row[mode] = {"rho_ranked": rho(m_nps, r_nps), "swap": swap_rate(m_nps, r_nps),
                         "rho_loud": rho(m_nps, loud), "rho_flux": rho(m_nps, flux),
                         "silence": silence_rate(shipped.notes, act),
                         "breaks": break_note_rate(shipped.notes, gaps)}
        rows.append(row)
        print(f"  {n:3d}/{len(chosen)} {band:16s} " + "  ".join(
            f"{m} {row[m]['rho_ranked']:+.2f}/{row[m]['swap']:.2f}" for m in args.modes)
            + f"   {row['map'][:50]}", flush=True)

    def mean(sel, key):
        v = [r[sel][key] for r in rows if sel in r and not np.isnan(r[sel][key])]
        return float(np.mean(v)) if v else np.nan

    labels = [b[2] for b in SR_BANDS]
    print(f"\n{len(rows)} maps. Per mode: rho vs ranked / swap / rho loud / rho flux / silence / breaks")
    print(f"  {'ranked':8s} {'':>23s} {mean('ranked', 'rho_loud'):+.3f} "
          f"{mean('ranked', 'rho_flux'):+.3f} {mean('ranked', 'silence'):.3f}")
    summary = {"step": step, "maps": len(rows), "ranked": {k: mean("ranked", k) for k in
                                                           ("rho_loud", "rho_flux", "silence")}}
    for mode in args.modes:
        keys = ("rho_ranked", "swap", "rho_loud", "rho_flux", "silence", "breaks")
        summary[mode] = {k: mean(mode, k) for k in keys}
        print(f"  {mode:8s} " + " ".join(f"{summary[mode][k]:+.3f}" for k in keys))
    print("\nby band, rho vs ranked (swap)")
    for label in labels:
        sub = [r for r in rows if r["band"] == label]
        print(f"  {label:16s} " + "  ".join(
            f"{m} {np.nanmean([r[m]['rho_ranked'] for r in sub]):+.2f} "
            f"({np.nanmean([r[m]['swap'] for r in sub]):.2f})" for m in args.modes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "maps": rows}, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
