"""High-precision wait utility for real-time control loops.

Uses sleep for the bulk of the wait period, then busy-spins
for the final ~1ms to achieve sub-millisecond timing accuracy.

From UMI's real-time control pattern.
"""

import time


def precise_wait(t_end: float, slack: float = 0.001, time_func=time.monotonic) -> None:
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
