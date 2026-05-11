"""Timestamped SE3 trajectory buffer for smooth robot control.

Stores predicted EE waypoints with target execution timestamps.
Provides time-based interpolation and SE3-aware chunk blending.

Patterns from:
  - lerobot/src/lerobot/async_inference (timestamp-based action timing, skip expired)
  - notes/controller_architecture_design.md (SE3 blending)
"""

import threading
import time
from collections import deque

import numpy as np


def _action_to_pose(pos: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Convert position + rotation vector to 4x4 pose matrix."""
    T = np.eye(4)
    T[:3, 3] = pos
    angle = np.linalg.norm(rotvec)
    if angle > 1e-10:
        axis = rotvec / angle
        c, s = np.cos(angle), np.sin(angle)
        v = 1 - c
        x, y, z = axis
        T[:3, :3] = np.array([
            [x*x*v + c,   x*y*v - z*s, x*z*v + y*s],
            [y*x*v + z*s, y*y*v + c,   y*z*v - x*s],
            [z*x*v - y*s, z*y*v + x*s, z*z*v + c],
        ])
    return T


def _pose_to_pos_rotvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and rotation vector from 4x4 pose."""
    pos = T[:3, 3].copy()
    R = T[:3, :3]
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-10:
        return pos, np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2 * np.sin(angle))
    return pos, axis * angle


def _blend_se3(T_old: np.ndarray, T_new: np.ndarray, alpha: float) -> np.ndarray:
    """Blend two SE3 transforms. alpha=0 → T_old, alpha=1 → T_new.

    Position: linear interpolation.
    Rotation: rotvec scaling (correct for small angles).
    """
    pos_old, rv_old = _pose_to_pos_rotvec(T_old)
    pos_new, rv_new = _pose_to_pos_rotvec(T_new)

    pos = (1 - alpha) * pos_old + alpha * pos_new
    rotvec = (1 - alpha) * rv_old + alpha * rv_new

    return _action_to_pose(pos, rotvec)


class TrajectoryBuffer:
    """Thread-safe timestamped SE3 trajectory buffer.

    Inference thread writes chunks via update().
    Control thread reads interpolated targets via interpolate().

    Each waypoint has:
      - 4x4 pose matrix (world frame)
      - gripper value (float)
      - target execution time (monotonic seconds)
    """

    def __init__(self, maxlen: int = 50):
        self._poses: deque[np.ndarray] = deque(maxlen=maxlen)
        self._grippers: deque[float] = deque(maxlen=maxlen)
        self._times: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._poses)

    def remaining(self) -> int:
        """Number of unexpired waypoints."""
        with self._lock:
            return len(self._poses)

    def update(self, world_poses_7d: np.ndarray, t_start: float | None = None,
               dt: float = 1.0 / 30.0, blend_steps: int = 3) -> int:
        """Add a new predicted chunk to the buffer.

        Args:
            world_poses_7d: (N, 7) array of [x, y, z, rx, ry, rz, gripper] in world frame.
            t_start: Target time for first action. If None, uses time.monotonic().
            dt: Time between consecutive actions (seconds).
            blend_steps: Number of steps to blend with remaining old chunk.

        Returns:
            Number of waypoints actually stored (after skipping expired).
        """
        if t_start is None:
            t_start = time.monotonic()

        now = time.monotonic()

        with self._lock:
            n = len(world_poses_7d)

            # Build new timestamps
            new_times = [t_start + i * dt for i in range(n)]

            # Skip expired actions (timestamps already in the past)
            first_valid = 0
            for i, t in enumerate(new_times):
                if t > now:
                    first_valid = i
                    break
            else:
                if n > 0 and new_times[-1] <= now:
                    # All expired
                    return 0

            # Build new waypoints (only valid ones)
            new_poses = []
            new_grippers = []
            new_ts = []
            for i in range(first_valid, n):
                p = world_poses_7d[i]
                new_poses.append(_action_to_pose(p[:3], p[3:6]))
                new_grippers.append(float(p[6]))
                new_ts.append(new_times[i])

            if not new_poses:
                return 0

            if len(self._poses) == 0:
                # First chunk — store directly
                self._poses.extend(new_poses)
                self._grippers.extend(new_grippers)
                self._times.extend(new_ts)
                return len(new_poses)

            # Blend remaining old with new
            n_old = len(self._poses)
            n_blend = min(blend_steps, n_old, len(new_poses))

            blended_poses = []
            blended_grippers = []
            blended_times = []

            for i in range(n_blend):
                alpha = (i + 1) / (n_blend + 1)
                T_blend = _blend_se3(self._poses[i], new_poses[i], alpha)
                g_blend = (1 - alpha) * self._grippers[i] + alpha * new_grippers[i]
                t_blend = (1 - alpha) * self._times[i] + alpha * new_ts[i]
                blended_poses.append(T_blend)
                blended_grippers.append(g_blend)
                blended_times.append(t_blend)

            # After blend window, use new chunk entirely
            for i in range(n_blend, len(new_poses)):
                blended_poses.append(new_poses[i])
                blended_grippers.append(new_grippers[i])
                blended_times.append(new_ts[i])

            # Replace buffer contents
            self._poses.clear()
            self._grippers.clear()
            self._times.clear()
            self._poses.extend(blended_poses)
            self._grippers.extend(blended_grippers)
            self._times.extend(blended_times)

            return len(blended_poses)

    def interpolate(self, t_now: float) -> tuple[np.ndarray, float] | None:
        """Get interpolated target for the given time.

        Returns (T_4x4, gripper) or None if buffer is empty.
        Discards expired waypoints before interpolating.
        """
        with self._lock:
            if len(self._poses) == 0:
                return None

            # Discard expired waypoints (keep at least one for extrapolation)
            while len(self._times) > 1 and self._times[0] < t_now:
                self._poses.popleft()
                self._grippers.popleft()
                self._times.popleft()

            if len(self._poses) == 0:
                return None

            if len(self._poses) == 1:
                return self._poses[0].copy(), self._grippers[0]

            # Interpolate between first two waypoints
            t0, t1 = self._times[0], self._times[1]
            if t1 <= t0:
                return self._poses[0].copy(), self._grippers[0]

            alpha = np.clip((t_now - t0) / (t1 - t0), 0.0, 1.0)
            T_blend = _blend_se3(self._poses[0], self._poses[1], alpha)
            g_blend = (1 - alpha) * self._grippers[0] + alpha * self._grippers[1]

            return T_blend, g_blend

    def clear(self) -> None:
        with self._lock:
            self._poses.clear()
            self._grippers.clear()
            self._times.clear()
