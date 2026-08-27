"""
taiko/data/motif.py

The canonical motif vector: a 16-dimensional description of *how* a stretch of
taiko is written, as opposed to how hard it is.

There is exactly one definition, here, used by preprocessing, training and
inference alike. Two divergent definitions previously existed -- a map-level one
in scripts/analyze_motifs.py that was written to the index and never read, and a
window-level one in the dataset that was actually trained on -- which meant the
"style" a user asked for at generation time was not the quantity the model had
learned.

    dim  meaning
    ---  ---------------------------------------------------------------
      0  IOI at 1/8 beat      fraction of gaps at this spacing
      1  IOI at 1/6 beat
      2  IOI at 1/4 beat
      3  IOI at 1/3 beat
      4  IOI at 1/2 beat
      5  IOI at 3/4 beat
      6  IOI at 1/1 beat
      7  kat fraction         share of hits that are kat (don is 1 - this)
      8  big fraction         share of hits that are big notes
      9  roll density         share of frames inside a drumroll
     10  denden density       share of frames inside a denden
     11  note density         onsets per second, normalised
     12  burst fraction       share of hits inside a run of 4+ at 1/4 or faster
     13  colour change rate   share of adjacent pairs that switch don <-> kat
     14  pattern entropy      Shannon entropy of don/kat 2-grams
     15  density variance     how much density swings across the window

Why this is quantised
---------------------
The training-time motif is measured on the very window the model is asked to
generate, which makes it very nearly a summary of the answer. Left continuous
and always present, the model learns to read the answer off the conditioning
vector: training loss looks excellent and generation collapses, because a user
supplies a rough hand-picked vector rather than an oracle.

Three defences, all applied here rather than at the call site so they cannot be
forgotten:

  * quantise every dimension to N_BUCKETS levels, so it carries a coarse
    intention rather than a precise measurement
  * drop individual dimensions independently during training, so no single
    dimension can be relied on
  * jitter the surviving dimensions

The ablation that proves it worked: generate with a motif borrowed from an
unrelated map. If quality collapses, the model is still leaking.
"""

from __future__ import annotations

import numpy as np

from taiko.data.frames import FRAME_MS
from taiko.data.tensor_repr import (
    CH_BIG_DON, CH_BIG_KAT, CH_DENDEN, CH_DON, CH_KAT, CH_ROLL,
    TM_COS, TM_SIN,
)

MOTIF_DIM = 16

MOTIF_NAMES = [
    "ioi_1_8", "ioi_1_6", "ioi_1_4", "ioi_1_3", "ioi_1_2", "ioi_3_4", "ioi_1_1",
    "kat_frac", "big_frac", "roll_density", "denden_density",
    "note_density", "burst_frac", "colour_change", "pattern_entropy",
    "density_variance",
]
assert len(MOTIF_NAMES) == MOTIF_DIM

# Inter-onset intervals, as fractions of a beat.
IOI_FRACTIONS = [1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 3 / 4, 1.0]
IOI_REL_TOL   = 0.06        # +/-6% of a beat

# Quantisation. 8 levels is coarse enough to stop the vector acting as a copy
# of the target and fine enough to express a real stylistic preference.
N_BUCKETS = 8

# Normalising constant for dim 11. 16 onsets/second is around the ceiling of
# human-playable taiko, so this maps the realistic range onto [0, 1].
MAX_NPS = 16.0

# A run this long at 1/4 or faster counts as a burst.
BURST_MIN_RUN = 4


# --------------------------------------------------------------------------- #
# Beat length recovery
# --------------------------------------------------------------------------- #

def beat_frames_from_timing(timing: np.ndarray) -> float:
    """
    Recover the beat length, in frames, from a timing stream.

    The sin/cos pair is a unit phasor advancing by a constant increment each
    frame, so the increment gives the period directly. Using the median makes
    this robust to a tempo change inside the window.

    Returns 0.0 when the stream carries no usable phase, which callers must
    treat as "beat length unknown" rather than substituting a guess.
    """
    if timing.shape[1] < 3:
        return 0.0

    phase = np.arctan2(
        timing[TM_SIN].astype(np.float64),
        timing[TM_COS].astype(np.float64),
    )
    if not np.any(np.abs(timing[TM_SIN]) > 1e-6) and not np.any(np.abs(timing[TM_COS]) > 1e-6):
        return 0.0

    # Wrap deltas into (-pi, pi] so the 2*pi rollover does not dominate.
    delta = np.diff(phase)
    delta = (delta + np.pi) % (2 * np.pi) - np.pi

    positive = delta[delta > 1e-9]
    if positive.size < 2:
        return 0.0

    step = float(np.median(positive))
    if step <= 1e-9:
        return 0.0

    return float(2 * np.pi / step)


def beat_frames_from_bpm(bpm: float) -> float:
    if bpm <= 0:
        return 0.0
    return (60_000.0 / bpm) / FRAME_MS


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def _onset_frames(channel: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Rising edges of a channel."""
    above = channel > threshold
    if not above.any():
        return np.empty(0, dtype=np.int64)
    prev = np.concatenate([[False], above[:-1]])
    return np.flatnonzero(above & ~prev)


def compute_motif(
    chart: np.ndarray,
    beat_frames: float,
    quantise: bool = True,
) -> np.ndarray:
    """
    Measure the motif of a chart window.

    Args:
        chart:       [6, W] float32
        beat_frames: beat length in frames, from `beat_frames_from_timing`.
                     When 0, the IOI and burst dimensions are left at zero
                     rather than computed against a fabricated tempo.
        quantise:    snap to N_BUCKETS levels. Always true for training and
                     inference; false is for analysis only.

    Returns:
        [16] float32 in [0, 1].
    """
    motif = np.zeros(MOTIF_DIM, dtype=np.float32)
    if chart.ndim != 2 or chart.shape[1] == 0:
        return motif

    width = chart.shape[1]
    window_sec = width * FRAME_MS / 1000.0

    don     = _onset_frames(chart[CH_DON])
    kat     = _onset_frames(chart[CH_KAT])
    big_don = _onset_frames(chart[CH_BIG_DON])
    big_kat = _onset_frames(chart[CH_BIG_KAT])

    # Colour, tracked alongside position so patterns can be read off later.
    # 0 = don-like, 1 = kat-like.
    positions = np.concatenate([don, kat, big_don, big_kat])
    colours   = np.concatenate([
        np.zeros(len(don), dtype=np.int8),
        np.ones(len(kat), dtype=np.int8),
        np.zeros(len(big_don), dtype=np.int8),
        np.ones(len(big_kat), dtype=np.int8),
    ])
    is_big = np.concatenate([
        np.zeros(len(don) + len(kat), dtype=bool),
        np.ones(len(big_don) + len(big_kat), dtype=bool),
    ])

    if positions.size == 0:
        return _finish(motif, quantise)

    order     = np.argsort(positions, kind="stable")
    positions = positions[order]
    colours   = colours[order]
    is_big    = is_big[order]
    total     = positions.size

    # --- dims 0-6: inter-onset interval histogram -------------------------- #
    iois = np.diff(positions).astype(np.float64) if total >= 2 else np.empty(0)

    if beat_frames > 0 and iois.size:
        tol = max(0.5, IOI_REL_TOL * beat_frames)
        for i, frac in enumerate(IOI_FRACTIONS):
            target = frac * beat_frames
            motif[i] = np.count_nonzero(np.abs(iois - target) <= tol) / iois.size

    # --- dims 7-8: note type mix ------------------------------------------- #
    motif[7] = float(np.count_nonzero(colours == 1)) / total
    motif[8] = float(np.count_nonzero(is_big)) / total

    # --- dims 9-10: sustained note density --------------------------------- #
    motif[9]  = float(np.count_nonzero(chart[CH_ROLL]   > 0.5)) / width
    motif[10] = float(np.count_nonzero(chart[CH_DENDEN] > 0.5)) / width

    # --- dim 11: overall density ------------------------------------------- #
    motif[11] = min(total / max(window_sec, 1e-6) / MAX_NPS, 1.0)

    # --- dim 12: bursts ---------------------------------------------------- #
    # Notes packed at 1/4 or tighter, in runs of at least BURST_MIN_RUN. This
    # is what separates a stream map from a map with the same average density
    # spread evenly.
    if beat_frames > 0 and iois.size:
        tight = iois <= beat_frames * (1 / 4) * (1 + IOI_REL_TOL)
        in_burst = np.zeros(total, dtype=bool)
        run_start = 0
        for i in range(len(tight) + 1):
            if i == len(tight) or not tight[i]:
                run_len = i - run_start + 1
                if run_len >= BURST_MIN_RUN:
                    in_burst[run_start:i + 1] = True
                run_start = i + 1
        motif[12] = float(np.count_nonzero(in_burst)) / total

    # --- dim 13: colour changes -------------------------------------------- #
    if total >= 2:
        motif[13] = float(np.count_nonzero(np.diff(colours) != 0)) / (total - 1)

    # --- dim 14: 2-gram entropy -------------------------------------------- #
    # Four possible don/kat pairs; entropy is normalised by log2(4) so a map
    # using all four equally scores 1.0 and a monotone one scores 0.
    if total >= 2:
        bigrams = colours[:-1].astype(np.int32) * 2 + colours[1:].astype(np.int32)
        counts  = np.bincount(bigrams, minlength=4).astype(np.float64)
        probs   = counts[counts > 0] / counts.sum()
        motif[14] = float(-(probs * np.log2(probs)).sum() / 2.0)

    # --- dim 15: density variance ------------------------------------------ #
    # Density measured over eight equal slices. A map with a quiet verse and a
    # dense chorus scores high; a uniformly busy one scores near zero.
    n_slices = 8
    if width >= n_slices:
        edges = np.linspace(0, width, n_slices + 1).astype(np.int64)
        per_slice = np.array([
            np.count_nonzero((positions >= edges[i]) & (positions < edges[i + 1]))
            for i in range(n_slices)
        ], dtype=np.float64)
        mean = per_slice.mean()
        if mean > 0:
            # Coefficient of variation, squashed into [0, 1].
            motif[15] = float(min(per_slice.std() / mean, 1.0))

    return _finish(motif, quantise)


def _finish(motif: np.ndarray, quantise: bool) -> np.ndarray:
    np.clip(motif, 0.0, 1.0, out=motif)
    return quantise_motif(motif) if quantise else motif


def quantise_motif(motif: np.ndarray, n_buckets: int = N_BUCKETS) -> np.ndarray:
    """Snap to the nearest of `n_buckets` evenly spaced levels in [0, 1]."""
    levels = n_buckets - 1
    return (np.round(np.clip(motif, 0.0, 1.0) * levels) / levels).astype(np.float32)


# --------------------------------------------------------------------------- #
# Training-time corruption
# --------------------------------------------------------------------------- #

def corrupt_motif(
    motif: np.ndarray,
    rng: np.random.Generator,
    dim_dropout: float = 0.3,
    jitter: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply per-dimension dropout and jitter, so the model cannot lean on any one
    dimension being present and accurate.

    Returns:
        (motif, mask) where mask is 1.0 for dimensions that survived. The mask
        is handed to the model so a dropped dimension reads as "unspecified"
        rather than as "this dimension is zero" -- those mean different things,
        and conflating them is what makes a user leaving a field blank look
        like a request for silence.
    """
    out  = motif.astype(np.float32).copy()
    mask = np.ones(MOTIF_DIM, dtype=np.float32)

    if dim_dropout > 0:
        dropped = rng.random(MOTIF_DIM) < dim_dropout
        out[dropped]  = 0.0
        mask[dropped] = 0.0

    if jitter > 0:
        live = mask > 0
        out[live] = np.clip(
            out[live] + rng.normal(0.0, jitter, size=int(live.sum())), 0.0, 1.0
        )

    return out, mask


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #

def _preset(**kwargs: float) -> np.ndarray:
    """Build a motif from named dimensions; everything unnamed stays at zero."""
    vec = np.zeros(MOTIF_DIM, dtype=np.float32)
    for name, value in kwargs.items():
        vec[MOTIF_NAMES.index(name)] = value
    return quantise_motif(vec)


# Named starting points, so nobody has to hand-author sixteen floats. Each is a
# plausible centre of its style rather than an extreme.
PRESETS: dict[str, np.ndarray] = {
    "simple": _preset(
        ioi_1_2=0.45, ioi_1_1=0.35, ioi_1_4=0.15,
        kat_frac=0.35, big_frac=0.06, note_density=0.15,
        colour_change=0.4, pattern_entropy=0.6, density_variance=0.3,
    ),
    "standard": _preset(
        ioi_1_4=0.45, ioi_1_2=0.3, ioi_1_1=0.12,
        kat_frac=0.42, big_frac=0.05, roll_density=0.05,
        note_density=0.3, burst_frac=0.25,
        colour_change=0.55, pattern_entropy=0.8, density_variance=0.35,
    ),
    "stream": _preset(
        ioi_1_4=0.75, ioi_1_2=0.12,
        kat_frac=0.45, big_frac=0.03,
        note_density=0.5, burst_frac=0.8,
        colour_change=0.6, pattern_entropy=0.9, density_variance=0.45,
    ),
    "speed": _preset(
        ioi_1_4=0.55, ioi_1_8=0.3,
        kat_frac=0.45, big_frac=0.02,
        note_density=0.75, burst_frac=0.85,
        colour_change=0.65, pattern_entropy=0.9, density_variance=0.4,
    ),
    "tech": _preset(
        ioi_1_4=0.3, ioi_1_6=0.25, ioi_1_3=0.2, ioi_1_2=0.15,
        kat_frac=0.48, big_frac=0.08, roll_density=0.04,
        note_density=0.4, burst_frac=0.35,
        colour_change=0.7, pattern_entropy=0.95, density_variance=0.6,
    ),
    "big_heavy": _preset(
        ioi_1_2=0.4, ioi_1_4=0.3, ioi_1_1=0.2,
        kat_frac=0.4, big_frac=0.3,
        note_density=0.22, burst_frac=0.1,
        colour_change=0.5, pattern_entropy=0.75, density_variance=0.3,
    ),
}


def get_preset(name: str) -> np.ndarray:
    if name not in PRESETS:
        raise KeyError(f"unknown motif preset {name!r}. Available: {sorted(PRESETS)}")
    return PRESETS[name].copy()


def describe_motif(motif: np.ndarray) -> str:
    """Human-readable dump, for inspecting what a reference map actually asks for."""
    return "\n".join(
        f"  {name:<17s} {value:.3f}"
        for name, value in zip(MOTIF_NAMES, motif)
    )
