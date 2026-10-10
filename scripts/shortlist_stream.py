"""
Shortlist ranked maps for the stream preset (ADR 0001): 5-7*, long 1/4 runs,
low tech share. Reads the shard index (ranked, md5-verified by construction)
and its note_times; nothing in Songs is touched.

    python scripts/shortlist_stream.py [--top 15]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from taiko.data.shards import ShardReader, load_note_times, decode_timing_points

DEATHSTREAM_SETS = {2283239, 1977770, 2009040, 2111683, 2111842, 2139170, 2159934, 2180835,
                    2191269, 2217406, 2287038, 2390049, 2422005, 2415871, 2447178, 2559071}
LONG_RUN = 16          # notes; a run this long at 1/4 counts as a stream


def gaps_in_beats(times, reds):
    """Each gap as a fraction of the beat of the red line it starts under."""
    starts = np.array([r.time for r in reds])
    mpb = np.array([r.beat_length for r in reds])
    k = np.clip(np.searchsorted(starts, times[:-1], side="right") - 1, 0, None)
    return np.diff(times) / mpb[k]


def near(g, frac):
    return np.abs(g - frac) <= max(0.02, 0.03 * frac)


def study(times, reds):
    g = gaps_in_beats(times.astype(float), reds)
    quarter = near(g, 1 / 4)
    tech = near(g, 1 / 3) | near(g, 1 / 6) | near(g, 1 / 12)
    in_run, run, longest = np.zeros(len(times), bool), 0, 0
    for i, q in enumerate(np.append(quarter, False)):
        if q:
            run += 1
            continue
        if run + 1 >= LONG_RUN:
            in_run[i - run:i + 1] = True
        longest, run = max(longest, run + 1), 0
    return {"q": quarter.mean(), "tech": tech.mean(), "eighth": near(g, 1 / 8).mean(),
            "run_share": in_run.mean(), "longest": longest}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/processed/shards"))
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    reader = ShardReader(args.shards)
    times = load_note_times(reader)
    rows = []
    for rec, t in zip(reader.records, times):
        sr = rec["difficulty"]
        if not 5.0 <= sr < 7.0 or rec["beatmapset_id"] in DEATHSTREAM_SETS or len(t) < 200:
            continue
        s = study(np.asarray(t), decode_timing_points(rec["timing_points"]))
        if s["tech"] > 0.03 or s["eighth"] > 0.02:
            continue
        rows.append((s["run_share"], rec, s))
    rows.sort(key=lambda r: -r[0])
    seen = set()
    print(f"{len(rows)} candidates (5-7*, tech <= 3%, 1/8 <= 2%, deathstream refs excluded)\n")
    print(f"{'run16%':>6} {'1/4%':>5} {'tech%':>5} {'long':>5} {'SR':>5} {'BPM':>4}  map")
    for share, rec, s in rows:
        if rec["beatmapset_id"] in seen:
            continue                       # one difficulty per set
        seen.add(rec["beatmapset_id"])
        print(f"{100*share:6.0f} {100*s['q']:5.0f} {100*s['tech']:5.1f} {s['longest']:5d} "
              f"{rec['difficulty']:5.2f} {rec['bpm']:4.0f}  {rec['title']} [{rec['version']}] "
              f"({rec['creator']}) {rec['split']}")
        if len(seen) == args.top:
            break


if __name__ == "__main__":
    main()
