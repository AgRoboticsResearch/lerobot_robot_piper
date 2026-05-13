"""Time translation utilities for multi-process controller.

Inference process uses time.time() (wall clock, convenient for logging).
Controller process uses time.monotonic() (never goes backward, guaranteed monotonic).

UMI uses the same pattern in RTDEInterpolationController:
    target_time = time.monotonic() - time.time() + target_time
"""

import time


def wall_to_monotonic(wall_t: float) -> float:
    """Convert wall clock time to monotonic time."""
    return wall_t - time.time() + time.monotonic()


def monotonic_to_wall(mono_t: float) -> float:
    """Convert monotonic time to wall clock time."""
    return mono_t - time.monotonic() + time.time()
