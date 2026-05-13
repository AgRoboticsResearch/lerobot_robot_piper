"""Immutable SE3 trajectory interpolator using linear position + SLERP rotation.

Direct port from UMI (universal_manipulation_interface/umi/common/pose_trajectory_interpolator.py).
Every operation returns a new instance — the interpolator is never mutated.

Pose format: (N, 6) arrays [x, y, z, rx, ry, rz] using rotation vectors.
"""

from __future__ import annotations

import numbers
from typing import Union

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()


def pose_distance(
    start_pose: np.ndarray, end_pose: np.ndarray
) -> tuple[float, float]:
    start_pose = np.asarray(start_pose)
    end_pose = np.asarray(end_pose)
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator:
    def __init__(
        self, times: np.ndarray, poses: np.ndarray
    ) -> None:
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times, dtype=np.float64)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses, dtype=np.float64)

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:, :3]
            rot = st.Rotation.from_rotvec(poses[:, 3:])

            self.pos_interp = si.interp1d(times, pos, axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:, :3] = self.pos_interp.y
            poses[:, 3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    @property
    def n_waypoints(self) -> int:
        return len(self.times)

    def __len__(self) -> int:
        return self.n_waypoints

    def __repr__(self) -> str:
        return (
            f"PoseTrajectoryInterpolator("
            f"n={self.n_waypoints}, "
            f"t=[{self.times[0]:.4f}..{self.times[-1]:.4f}])"
        )

    def trim(
        self, start_t: float, end_t: float
    ) -> PoseTrajectoryInterpolator:
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)

    def drive_to_waypoint(
        self,
        pose: np.ndarray,
        time: float,
        curr_time: float,
        max_pos_speed: float = float("inf"),
        max_rot_speed: float = float("inf"),
    ) -> PoseTrajectoryInterpolator:
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        time = max(time, curr_time)

        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(
        self,
        pose: np.ndarray,
        time: float,
        max_pos_speed: float = float("inf"),
        max_rot_speed: float = float("inf"),
        curr_time: float | None = None,
        last_waypoint_time: float | None = None,
    ) -> PoseTrajectoryInterpolator:
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        assert start_time <= end_time
        assert end_time <= time
        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)

        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def __call__(
        self, t: Union[numbers.Number, np.ndarray]
    ) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose
