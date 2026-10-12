"""--fit-hours: the cosine horizon ends as the session's budget does."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from train_diffusion import FIT_STEP_SHARE, fitted_horizon  # noqa: E402

# 220 steps in, 4 s/step, 900 s spent of a 10 h budget: the rest at 4 s/step.
total = fitted_horizon(220, 4.0, 900.0, 36_000.0)
assert total == 220 + int((36_000 - 900) * FIT_STEP_SHARE / 4.0), total
# Twice as slow, about half the steps left.
assert abs(fitted_horizon(220, 8.0, 900.0, 36_000.0) - 220 - (total - 220) / 2) <= 1
# Budget already spent: no more steps, never fewer than done.
assert fitted_horizon(220, 4.0, 40_000.0, 36_000.0) == 220
print("all fit-hours tests passed")
