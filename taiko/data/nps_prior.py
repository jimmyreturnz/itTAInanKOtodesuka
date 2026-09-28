"""
taiko/data/nps_prior.py

Star rating -> typical (avg_nps, peak_nps), fitted on the training maps, so
generation can ask for a realistic density instead of zero. Stored as plain
lists so it pickles into a checkpoint and round-trips through JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

BIN_WIDTH = 0.5
MIN_MAPS = 5


def fit_nps_prior(records: Iterable[dict]) -> Optional[dict]:
    """Median avg/peak NPS per 0.5* bin. None if no record has both."""
    rows = [(float(r.get("difficulty", 0)), float(r.get("avg_nps", 0)), float(r.get("peak_nps", 0)))
            for r in records]
    rows = np.asarray([r for r in rows if r[0] > 0 and r[1] > 0 and r[2] > 0])
    if rows.size == 0:
        return None

    bins = np.floor(rows[:, 0] / BIN_WIDTH).astype(int)
    stars, avg, peak, count = [], [], [], []
    for b in np.unique(bins):
        sel = rows[bins == b]
        if len(sel) < MIN_MAPS:
            continue
        stars.append(float(np.median(sel[:, 0])))
        avg.append(float(np.median(sel[:, 1])))
        peak.append(float(np.median(sel[:, 2])))
        count.append(int(len(sel)))
    if not stars:  # too few maps for any bin: one global row
        stars, avg, peak = ([float(np.median(rows[:, i]))] for i in range(3))
        count = [int(len(rows))]
    return {"stars": stars, "avg_nps": avg, "peak_nps": peak, "count": count}


def lookup(prior: dict, star_rating: float) -> tuple[float, float]:
    """Interpolated between bins, clamped at the ends."""
    x = prior["stars"]
    return (float(np.interp(star_rating, x, prior["avg_nps"])),
            float(np.interp(star_rating, x, prior["peak_nps"])))


def save_prior(prior: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(prior, indent=2))


def load_prior(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def describe(prior: Optional[dict]) -> str:
    if not prior:
        return "NPS prior: no maps with star rating and density"
    lines = ["  stars   avg nps  peak nps   maps"]
    for s, a, p, c in zip(prior["stars"], prior["avg_nps"], prior["peak_nps"], prior["count"]):
        lines.append(f"  {s:5.2f}  {a:8.2f}  {p:8.2f}  {c:5d}")
    return "\n".join(lines)


if __name__ == "__main__":
    recs = [{"difficulty": 2 + i % 6, "avg_nps": 1.0 + i % 6, "peak_nps": 2.0 + 2 * (i % 6)}
            for i in range(60)]
    pr = fit_nps_prior(recs)
    assert lookup(pr, 2.0) == (1.0, 2.0)
    assert lookup(pr, 2.5) == (1.5, 3.0)
    assert lookup(pr, 99) == (6.0, 12.0)
    assert fit_nps_prior([]) is None
    assert fit_nps_prior(recs[:3])["count"] == [3]
    print(describe(pr))
    print("nps_prior ok")
