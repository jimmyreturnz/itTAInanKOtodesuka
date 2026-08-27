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
