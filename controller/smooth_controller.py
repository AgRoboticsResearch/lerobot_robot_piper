"""Smooth controller: background thread running IK at fixed frequency.

Reads EE targets from TrajectoryBuffer, solves IK, applies velocity
limits, and sends joint commands to the Piper robot arm.

Pattern from:
  - lerobot/src/lerobot/robots/unitree_g1 (dual-thread with Lock)
  - lerobot/src/lerobot/async_inference/robot_client (action buffer consumption)
"""

import logging
import threading
import time

import numpy as np

from .precise_wait import precise_wait
from .trajectory_buffer import TrajectoryBuffer

logger = logging.getLogger(__name__)

JOINT_LIMITS_DEG = {
    "min": np.array([-150.0, 0.0, -170.0, -100.0, -70.0, -120.0]),
    "max": np.array([150.0, 180.0, 0.0, 100.0, 70.0, 120.0]),
}
JOINT_LIMIT_TOLERANCE_DEG = 0.5


class SmoothController:
    """Background IK control thread that executes trajectories from TrajectoryBuffer.

    Usage:
        ctrl = SmoothController(piper_robot, kin, frequency=50)
        ctrl.start()
        ctrl.buffer.update(pred_world, t_start=..., dt=1/30)
        ...
        ctrl.stop()
    """

    def __init__(
        self,
        piper_robot,
        kin,
        frequency: float = 50.0,
        max_vel_deg_s: float = 60.0,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ):
        self.piper = piper_robot
        self.kin = kin
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.max_vel_deg_s = max_vel_deg_s
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight

        self.buffer = TrajectoryBuffer(maxlen=60)

        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0
        self._last_joints: np.ndarray | None = None
        self._last_gripper: float = 1.0

        # Stats for logging
        self._ik_errors = 0
        self._ik_total = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(target=self._control_loop, daemon=True, name="controller")
        self._thread.start()
        logger.info(f"Controller started at {self.frequency}Hz")

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info(f"Controller stopped (ran {self._tick_count} ticks, "
                     f"IK errors {self._ik_errors}/{self._ik_total})")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _get_current_joints(self) -> np.ndarray:
        """Read current joint positions from Piper."""
        obs = self.piper.get_observation()
        joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
        self._last_gripper = obs.get("gripper.pos", self._last_gripper)
        return joints

    def _velocity_limit(self, target_joints: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """Clamp joint velocities to max_vel_deg_s."""
        max_step = self.max_vel_deg_s * self.dt
        error = target_joints - current_joints
        max_error = np.max(np.abs(error))
        if max_error > max_step:
            scale = max_step / max_error
            return current_joints + error * scale
        return target_joints

    def _joint_limit_check(self, joints: np.ndarray) -> np.ndarray:
        """Clamp joints to within limits."""
        lo = JOINT_LIMITS_DEG["min"] - JOINT_LIMIT_TOLERANCE_DEG
        hi = JOINT_LIMITS_DEG["max"] + JOINT_LIMIT_TOLERANCE_DEG
        return np.clip(joints, lo, hi)

    def _send_action(self, joints: np.ndarray, gripper: float) -> None:
        """Send joint command to Piper (arm via CAN, gripper via serial)."""
        action = {f"joint_{i+1}.pos": joints[i] for i in range(6)}
        action["gripper.pos"] = gripper
        self.piper.send_action(action)

    def _control_loop(self) -> None:
        t_start = time.monotonic()

        # Initialize with current joints
        try:
            self._last_joints = self._get_current_joints()
        except Exception as e:
            logger.error(f"Controller: failed to read initial joints: {e}")
            return

        last_log_time = t_start

        while not self._shutdown.is_set():
            tick = self._tick_count
            t_target = t_start + tick * self.dt

            try:
                target = self.buffer.interpolate(t_target)

                if target is not None:
                    T_target, gripper = target
                    self._ik_total += 1

                    try:
                        joints_target = self.kin.inverse_kinematics(
                            self._last_joints, T_target,
                            position_weight=self.position_weight,
                            orientation_weight=self.orientation_weight,
                        )
                    except Exception as e:
                        self._ik_errors += 1
                        if self._ik_errors <= 5:
                            logger.warning(f"Controller: IK failed: {e}")
                    else:
                        joints_safe = self._joint_limit_check(joints_target)
                        joints_cmd = self._velocity_limit(joints_safe, self._last_joints)

                        try:
                            self._send_action(joints_cmd, gripper)
                            self._last_joints = joints_cmd.copy()
                            self._last_gripper = gripper
                        except Exception as e:
                            if self._ik_errors <= 5:
                                logger.warning(f"Controller: send_action failed: {e}")
                else:
                    # No target — hold position
                    pass

            except Exception as e:
                logger.error(f"Controller tick error: {e}")

            self._tick_count += 1

            # Periodic logging
            now = time.monotonic()
            if now - last_log_time >= 2.0:
                buf_len = len(self.buffer)
                logger.debug(f"Controller: tick={tick} buffer={buf_len} "
                              f"IK_errors={self._ik_errors}/{self._ik_total}")
                last_log_time = now

            precise_wait(t_start + self._tick_count * self.dt)

    def hold_position(self) -> None:
        """Hold current position for 3 ticks."""
        if self._last_joints is not None:
            for _ in range(3):
                try:
                    self._send_action(self._last_joints, self._last_gripper)
                except Exception:
                    pass
                time.sleep(0.02)

    def get_stats(self) -> dict:
        return {
            "ticks": self._tick_count,
            "ik_errors": self._ik_errors,
            "ik_total": self._ik_total,
            "buffer_remaining": len(self.buffer),
        }
