"""Precise timing utilities for real-time control loops.

Direct port from UMI (diffusion_policy/common/precise_sleep.py).
Uses hybrid of time.sleep and busy-spin to minimize jitter.
All functions default to time.monotonic — never goes backward.
"""

import time


def precise_sleep(
    dt: float, slack: float = 0.001, time_func=time.monotonic
) -> None:
    """Sleep for exactly ``dt`` seconds with sub-ms precision.

    Args:
        dt: Duration to sleep (seconds).
        slack: Time to reserve for busy-spin (seconds). Default 1ms.
        time_func: Clock function. Default time.monotonic.
    """
    t_start = time_func()
    if dt > slack:
        time.sleep(dt - slack)
    t_end = t_start + dt
    while time_func() < t_end:
        pass


def precise_wait(
    t_end: float, slack: float = 0.001, time_func=time.monotonic
) -> None:
    """Wait until ``t_end`` with sub-ms precision.

    Args:
        t_end: Target time (monotonic clock).
        slack: Time to reserve for busy-spin (seconds). Default 1ms.
        time_func: Clock function. Default time.monotonic.
    """
    t_wait = t_end - time_func()
    if t_wait > 0:
        t_sleep = t_wait - slack
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
