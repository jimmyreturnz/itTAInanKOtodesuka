"""
taiko/data/conditioning.py

Constants shared by the data pipeline and the model for every conditioning
signal. Both sides must agree exactly: a normalisation constant that differs
between preprocessing and the model is a silent train/inference skew.

The null style
--------------
Style 0 is "standard", a real class. Using it as the unconditional token -- as
the previous classifier-free-guidance code did -- makes the unconditional branch
identical to the standard-style branch, so guidance pushes samples *away from
standard style* rather than away from unconditionality. STYLE_NULL is a
dedicated extra index that means "no style requested".
"""

from __future__ import annotations

import numpy as np

from taiko.eval.criteria import LEVELS

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

STYLE_NAMES = {
    0: "standard",
    1: "stream",
    2: "speed",
    3: "tech",
}

N_REAL_STYLES = len(STYLE_NAMES)
STYLE_NULL    = N_REAL_STYLES            # 4
N_STYLES      = N_REAL_STYLES + 1        # 5, including the null token

STYLE_TO_INT = {name: idx for idx, name in STYLE_NAMES.items()}


def style_to_int(name: str) -> int:
    """Style name -> index. Unknown names fall back to the null token."""
    return STYLE_TO_INT.get(name.lower().strip(), STYLE_NULL)


def style_to_name(idx: int) -> str:
    return STYLE_NAMES.get(int(idx), "unconditional")


# --------------------------------------------------------------------------- #
# Scalar normalisation
# --------------------------------------------------------------------------- #
# Each scalar is divided by its constant before reaching the model, so all
# conditioning inputs land in roughly [0, 1].

DIFFICULTY_SCALE = 10.0     # osu!taiko star ratings run to about 10
AVG_NPS_SCALE    = 12.0     # sustained notes per second
PEAK_NPS_SCALE   = 20.0     # peak over a 5 s window


def normalise_difficulty(sr: float) -> float:
    return max(0.0, min(float(sr) / DIFFICULTY_SCALE, 1.5))


def normalise_avg_nps(nps: float) -> float:
    return max(0.0, min(float(nps) / AVG_NPS_SCALE, 1.5))


def normalise_peak_nps(nps: float) -> float:
    return max(0.0, min(float(nps) / PEAK_NPS_SCALE, 1.5))


# Star-rating bands, shared by the eval (stratified pool, per-band scores) and
# training's SR-band sampling, so both mean the same buckets.
SR_BANDS = [(0, 2.0, "kantan <2*"), (2.0, 3.0, "futsuu 2-3*"), (3.0, 4.0, "muzukashii 3-4*"),
            (4.0, 5.5, "oni 4-5.5*"), (5.5, 7.0, "inner 5.5-7*"), (7.0, 99, "extreme 7*+")]


def sr_band(sr: float) -> int:
    return next(i for i, (lo, hi, _) in enumerate(SR_BANDS) if sr < hi or i == len(SR_BANDS) - 1)


# --------------------------------------------------------------------------- #
# Classifier-free guidance
# --------------------------------------------------------------------------- #

# Fraction of training samples that see the null condition. Standard practice is
# 0.1-0.2; the previous 0.5 halved the gradient reaching the conditional path.
CFG_DROPOUT = 0.15

# Independent dropout per motif dimension, on top of the all-or-nothing drop
# above. See taiko/data/motif.py for why this is not optional.
MOTIF_DIM_DROPOUT = 0.3
MOTIF_JITTER      = 0.05

# Density context (models built with features >= 2). Each is dropped
# independently so every combination of "given / unspecified" is trained, which
# is what lets generation leave either one out without asking for zero.
MAP_NPS_DROPOUT    = 0.2
WINDOW_NPS_DROPOUT = 0.3


# --------------------------------------------------------------------------- #
# Difficulty profile (features >= 3, ADR 0001)
# --------------------------------------------------------------------------- #
# The map-wide profile replaces style and the per-window motif. Its motif part
# travels in the old motif slot, computed over the whole map; the rest is this
# vector: tech share and 1/8 share, each with a known flag, the criteria level
# one-hot (all zero = unspecified), then the divisor mask: which snaps the map
# puts notes on, and a known flag. The mask tells the model when a song has no
# ternary at all, which is where the snap flicker comes from.

DIVISORS = (1, 2, 3, 4, 6, 8)
# Finer and odd snaps fold into the family they subdivide; 1/5, 1/7 and 1/9
# are left out (0.13% of 7*+ notes).
DIVISOR_FOLD = {1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 8: 8, 12: 6, 16: 8}
DIVISOR_MIN_SHARE = 0.005    # a snap counts as used above 0.5% of notes ...
DIVISOR_MIN_NOTES = 4        # ... and 4 notes, so one stray note does not open it

TECH, TECH_KNOWN, SPEED, SPEED_KNOWN, LEVEL0 = 0, 1, 2, 3, 4
DIV0 = LEVEL0 + len(LEVELS)
DIV_KNOWN = DIV0 + len(DIVISORS)
PROFILE_DIM = DIV_KNOWN + 1

# Each part is unspecified this often in training, so generation may omit it.
PROFILE_DROPOUT = 0.2


def divisors_used(snap_counts: dict) -> set[int]:
    """The DIVISORS a map puts notes on, from the pack's per-snap note counts."""
    folded: dict[int, int] = {}
    for d, n in snap_counts.items():
        if int(d) in DIVISOR_FOLD:
            folded[DIVISOR_FOLD[int(d)]] = folded.get(DIVISOR_FOLD[int(d)], 0) + n
    total = sum(snap_counts.values())
    return {d for d, n in folded.items()
            if n >= DIVISOR_MIN_NOTES and n >= DIVISOR_MIN_SHARE * total}


def profile_vector(tech_share: float | None = None, speed_share: float | None = None,
                   level: str | None = None, divisors: set[int] | None = None):
    """[PROFILE_DIM] float32. None leaves that part unspecified."""
    v = np.zeros(PROFILE_DIM, dtype=np.float32)
    if tech_share is not None:
        v[TECH], v[TECH_KNOWN] = min(max(float(tech_share), 0.0), 1.0), 1.0
    if speed_share is not None:
        v[SPEED], v[SPEED_KNOWN] = min(max(float(speed_share), 0.0), 1.0), 1.0
    if level in LEVELS:
        v[LEVEL0 + LEVELS.index(level)] = 1.0
    if divisors is not None:
        for d in divisors:
            v[DIV0 + DIVISORS.index(d)] = 1.0
        v[DIV_KNOWN] = 1.0
    return v


def record_profile(record: dict):
    """A packed map's own profile vector; parts it does not carry stay unspecified."""
    counts = record.get("snap_counts")
    return profile_vector(record.get("tech_share"), record.get("snap_1_8"), record.get("level"),
                          divisors_used(counts) if counts else None)
