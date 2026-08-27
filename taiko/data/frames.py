"""
taiko/data/frames.py

Single source of truth for the time grid.

Everything downstream -- mel extraction, chart tensors, dataset windowing,
latent lengths, inference -- derives its frame rate from this module. Nothing
else is allowed to hardcode a hop length, a frames-per-second, or a ratio
between the audio and chart grids.

    THE CONTRACT
    ------------
    One mel frame == one chart frame == FRAME_MS milliseconds.

    mel[:, i]  and  chart[:, i]  describe the same instant of the same song.

This is not a stylistic preference. Violating it silently destroys the model:
the U-Net still trains, the loss still falls, and the result is a generator
that has learned the marginal distribution of taiko rhythms while ignoring the
audio entirely. Use `assert_aligned()` at every boundary where a mel and a
chart tensor meet.

Derivation:
    FRAME_MS = HOP_LENGTH / SAMPLE_RATE * 1000
             = 441 / 22050 * 1000
             = 20.0 ms  ->  50 frames per second
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Audio grid
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 22_050        # Hz
HOP_LENGTH  = 441           # samples per frame

# --------------------------------------------------------------------------- #
# Derived time grid -- shared by audio and charts
# --------------------------------------------------------------------------- #

FRAME_MS  = HOP_LENGTH / SAMPLE_RATE * 1000.0    # 20.0
FRAME_SEC = FRAME_MS / 1000.0                    # 0.02
FPS       = 1000.0 / FRAME_MS                    # 50.0

# Audio and chart grids are identical. This constant exists so that call sites
# read as an explicit statement of the contract rather than a bare `1`.
MEL_FRAMES_PER_CHART_FRAME = 1

assert abs(FRAME_MS - 20.0) < 1e-9, "frame grid drifted from 20 ms"


# --------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------- #

def ms_to_frame(ms: float) -> int:
    """Milliseconds -> nearest frame index."""
    return int(round(ms / FRAME_MS))


def frame_to_ms(frame: int | float) -> float:
    """Frame index -> milliseconds at the start of that frame."""
    return frame * FRAME_MS


def sec_to_frames(seconds: float) -> int:
    """Duration in seconds -> number of frames."""
    return int(round(seconds * FPS))


def frames_to_sec(frames: int) -> float:
    """Number of frames -> duration in seconds."""
    return frames / FPS


def samples_to_frames(n_samples: int) -> int:
    """Waveform sample count -> mel frame count (matches STFT centre padding)."""
    return n_samples // HOP_LENGTH + 1


# --------------------------------------------------------------------------- #
# Contract enforcement
# --------------------------------------------------------------------------- #

class FrameAlignmentError(AssertionError):
    """Raised when a mel and a chart tensor do not share the same time grid."""


def assert_aligned(
    mel_frames: int,
    chart_frames: int,
    *,
    tolerance_frames: int = 100,
    context: str = "",
) -> None:
    """
    Fail loudly when a mel and a chart tensor disagree about the time grid.

    A chart tensor ends shortly after its last note while the mel runs to the
    end of the audio, so the mel is legitimately *longer* -- often by a lot,
    for maps that stop before the outro. What is never legitimate is the mel
    being materially *shorter* than the chart, or the two differing by a
    suspiciously exact factor.

    Args:
        tolerance_frames: how far the chart may overrun the mel before this is
            treated as a real misalignment rather than rounding at the tail.

    Raises:
        FrameAlignmentError: with a diagnosis, not just a failed comparison.
    """
    where = f" ({context})" if context else ""

    if chart_frames > mel_frames + tolerance_frames:
        ratio = chart_frames / max(mel_frames, 1)
        hint = ""
        if 1.8 < ratio < 2.2:
            hint = (
                "\n  The chart is ~2x the mel. Something is treating the chart "
                "grid as twice the audio grid."
            )
        raise FrameAlignmentError(
            f"chart runs past the end of the audio{where}:\n"
            f"  mel   frames = {mel_frames:>8d}  ({frames_to_sec(mel_frames):.1f}s)\n"
            f"  chart frames = {chart_frames:>8d}  ({frames_to_sec(chart_frames):.1f}s)\n"
            f"  ratio        = {ratio:.3f}"
            f"{hint}\n"
            f"  Both grids must be {FRAME_MS:g} ms/frame -- see taiko/data/frames.py"
        )

    ratio = mel_frames / max(chart_frames, 1)
    if 1.8 < ratio < 2.2 and chart_frames > 500:
        raise FrameAlignmentError(
            f"mel is ~2x the chart{where}: mel={mel_frames}, chart={chart_frames}.\n"
            f"  That is the classic symptom of a 10 ms mel paired with a 20 ms "
            f"chart, or of a window cropper multiplying an index by 2.\n"
            f"  Both grids must be {FRAME_MS:g} ms/frame -- see taiko/data/frames.py"
        )


def describe() -> str:
    """One-line summary for training logs."""
    return (
        f"time grid: {FRAME_MS:g} ms/frame ({FPS:g} fps), "
        f"hop={HOP_LENGTH} @ {SAMPLE_RATE} Hz, mel:chart = 1:1"
    )
