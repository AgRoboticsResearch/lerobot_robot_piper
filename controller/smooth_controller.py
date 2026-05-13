"""Smooth controller: separate process running IK at fixed frequency.

Mirrors UMI's RTDEInterpolationController architecture:
  - Runs as mp.Process (GIL-free, process isolation)
  - Commands via SharedMemoryQueue (lock-free, cross-process)
  - State via SharedMemoryRingBuffer (lock-free feedback)
  - Time: monotonic inside process, wall clock from external
  - PoseTrajectoryInterpolator: immutable, replaced on each command
"""

import enum
import logging
import multiprocessing as mp
import time
from queue import Empty

import numpy as np
from multiprocessing.managers import SharedMemoryManager
from scipy.spatial.transform import Rotation

from .pose_trajectory_interpolator import PoseTrajectoryInterpolator
from .precise_wait import precise_wait
from .shared_memory import SharedMemoryQueue, SharedMemoryRingBuffer
from .time_utils import wall_to_monotonic

logger = logging.getLogger(__name__)

JOINT_LIMITS_DEG = {
    "min": np.array([-150.0, 0.0, -170.0, -100.0, -70.0, -120.0]),
    "max": np.array([150.0, 180.0, 0.0, 100.0, 70.0, 120.0]),
}
JOINT_LIMIT_TOLERANCE_DEG = 0.5


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_JOINTS = 3


class SmoothController(mp.Process):
    """Background IK control process — UMI RTDEInterpolationController pattern.

    Usage:
        with SharedMemoryManager() as shm:
            ctrl = SmoothController(
                shm_manager=shm,
                piper_config={...},
                urdf_path="...",
                frequency=50,
            )
            ctrl.start()
            ctrl.start_wait()
            ctrl.schedule_waypoint(pose_6d, target_time)
            ...
            ctrl.stop()
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        piper_config: dict,
        urdf_path: str,
        target_frame: str = "ee_link",
        joint_names: list[str] | None = None,
        frequency: float = 50.0,
        max_vel_deg_s: float = 60.0,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
        max_pos_speed: float = float("inf"),
        max_rot_speed: float = float("inf"),
        launch_timeout: float = 3.0,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        super().__init__(name="PiperController")
        # Config (serialized — subprocess recreates objects from these)
        self.piper_config = piper_config
        self.urdf_path = urdf_path
        self.target_frame = target_frame
        self.joint_names = joint_names or [f"joint_{i+1}" for i in range(6)]
        self.frequency = frequency
        self.max_vel_deg_s = max_vel_deg_s
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.dry_run = dry_run

        # --- Shared memory IPC (created here, used in both processes) ---
        # Command queue: inference → controller
        cmd_example = {
            "cmd": np.int64(0),
            "target_pose": np.zeros(6, dtype=np.float64),
            "target_time": np.float64(0.0),
            "duration": np.float64(0.0),
            "gripper": np.float64(1.0),
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=cmd_example,
            buffer_size=256,
        )

        # State ring buffer: controller → inference/viz
        state_example = {
            "ActualJointState": np.zeros(6, dtype=np.float64),
            "ActualEEPose": np.zeros(6, dtype=np.float64),
            "gripper": np.float64(1.0),
            "robot_timestamp": np.float64(0.0),
            "robot_timestamp_mono": np.float64(0.0),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=state_example,
            get_max_k=int(frequency * 5),
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()

    # ========== Lifecycle (same as UMI) ==========

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop(self, wait=True):
        self.input_queue.put({"cmd": np.int64(Command.STOP.value)})
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========== Command interface (same as UMI RTDE controller) ==========

    def move_to_pose(self, pose_6d, duration: float = 0.1) -> None:
        """Send single EE target pose. Duration = desired time to reach pose."""
        self.input_queue.put(
            {
                "cmd": np.int64(Command.SERVOL.value),
                "target_pose": np.asarray(pose_6d, dtype=np.float64),
                "duration": np.float64(duration),
            }
        )

    def move_to_joints(
        self,
        joint_targets_deg: np.ndarray,
        duration: float = 3.0,
        gripper: float | None = None,
    ) -> None:
        """Move to target joint angles directly (no IK, joint-space interpolation)."""
        self.input_queue.put(
            {
                "cmd": np.int64(Command.MOVE_JOINTS.value),
                "target_pose": np.asarray(joint_targets_deg, dtype=np.float64),
                "duration": np.float64(duration),
                "gripper": np.float64(gripper if gripper is not None else 1.0),
            }
        )

    def schedule_waypoint(
        self,
        pose_6d: np.ndarray,
        target_time: float,
        gripper: float | None = None,
    ) -> None:
        """Schedule waypoint. target_time is wall clock (time.time())."""
        self.input_queue.put(
            {
                "cmd": np.int64(Command.SCHEDULE_WAYPOINT.value),
                "target_pose": np.asarray(pose_6d, dtype=np.float64),
                "target_time": np.float64(target_time),
                "gripper": np.float64(gripper if gripper is not None else 1.0),
            }
        )

    def exec_actions(
        self,
        actions: np.ndarray,
        obs_timestamps: float | None = None,
        dt: float = 1.0 / 30.0,
    ) -> int:
        """Execute predicted actions as schedule_waypoint commands.

        Mirrors UMI's RealEnv.exec_actions():
            action_timestamps = arange(N)*dt + obs_timestamps[-1]
            is_new = action_timestamps > receive_time
            schedule_waypoint for each new action

        Args:
            actions: (N, 7) array of [x, y, z, rx, ry, rz, gripper].
            obs_timestamps: Observation timestamp (time.time()). If None, uses time.time().
                If monotonic (small number), auto-converts to wall clock.
            dt: Time between consecutive actions.

        Returns:
            Number of non-expired actions scheduled.
        """
        if obs_timestamps is None:
            obs_timestamps = time.time()
        else:
            # If obs_timestamps looks like monotonic (small number), convert to wall
            if obs_timestamps < 1e9:
                obs_timestamps = obs_timestamps - time.monotonic() + time.time()

        # UMI pattern: action_timestamps = arange(N)*dt + obs_timestamps[-1]
        action_timestamps = np.arange(len(actions), dtype=np.float64) * dt + obs_timestamps

        # Filter expired actions
        receive_time = time.time()
        is_new = action_timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = action_timestamps[is_new]

        # Schedule waypoints
        for i in range(len(new_actions)):
            self.schedule_waypoint(
                new_actions[i, :6],
                new_timestamps[i],
                gripper=float(new_actions[i, 6]),
            )

        return len(new_actions)

    # ========== State feedback (same as UMI) ==========

    def get_state(self, k=None):
        if k is None:
            return self.ring_buffer.get()
        else:
            return self.ring_buffer.get_last_k(k=k)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def remaining(self) -> int:
        return self.input_queue.qsize()

    # ========== Main loop (runs in subprocess, mirrors UMI) ==========

    def run(self):
        # --- Create hardware connections in subprocess ---
        from lerobot_robot_piper.piper import Piper
        from lerobot_robot_piper.config_piper import PiperConfig
        from lerobot.model.kinematics import RobotKinematics

        piper = Piper(PiperConfig(**self.piper_config))
        piper.connect()

        if self.dry_run:
            try:
                piper._iface.piper.DisablePiper()
                if self.verbose:
                    logger.info("Controller: dry_run mode — motors disabled, no send_action")
            except Exception:
                pass

        kin = RobotKinematics(
            urdf_path=self.urdf_path,
            target_frame_name=self.target_frame,
            joint_names=self.joint_names,
        )
        dt = 1.0 / self.frequency

        if self.verbose:
            logger.info("Controller process: hardware connected")

        # --- Read initial state ---
        obs = piper.get_observation()
        last_joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
        last_gripper = obs.get("gripper.pos", 1.0)
        grip_value = last_gripper

        # --- Initialize interpolator at current EE pose (UMI pattern) ---
        T_curr = kin.forward_kinematics(last_joints)
        curr_pose_6d = np.concatenate(
            [T_curr[:3, 3], Rotation.from_matrix(T_curr[:3, :3]).as_rotvec()]
        )
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t], poses=[curr_pose_6d]
        )

        t_start = time.monotonic()
        iter_idx = 0
        keep_running = True
        ik_errors = 0
        ik_total = 0
        sensor_failures = 0

        # Joint-space move state (active when MOVE_JOINTS command received)
        joint_move_active = False
        joint_move_start: np.ndarray | None = None
        joint_move_end: np.ndarray | None = None
        joint_move_t_start: float = 0.0
        joint_move_t_end: float = 0.0
        joint_move_gripper: float = 1.0

        while keep_running:
            t_now = time.monotonic()

            if joint_move_active:
                # --- Joint-space interpolation (no IK) ---
                alpha = (t_now - joint_move_t_start) / max(
                    joint_move_t_end - joint_move_t_start, 1e-6
                )
                alpha = min(alpha, 1.0)
                joints_cmd = joint_move_start * (1 - alpha) + joint_move_end * alpha
                grip_value = joint_move_gripper

                action = {f"joint_{i+1}.pos": joints_cmd[i] for i in range(6)}
                action["gripper.pos"] = grip_value
                if not self.dry_run:
                    piper.send_action(action)

                if alpha >= 1.0:
                    joint_move_active = False
                    # Re-anchor the interpolator at the home pose so
                    # the IK loop doesn't pull us back to the old position.
                    last_joints = joint_move_end.copy()
                    T_new = kin.forward_kinematics(last_joints)
                    new_pose_6d = np.concatenate([
                        T_new[:3, 3],
                        Rotation.from_matrix(T_new[:3, :3]).as_rotvec(),
                    ])
                    curr_t = time.monotonic()
                    last_waypoint_time = curr_t
                    pose_interp = PoseTrajectoryInterpolator(
                        times=[curr_t], poses=[new_pose_6d]
                    )
                    if self.verbose:
                        logger.info("Controller: joint move complete, "
                                     f"interpolator re-anchored to "
                                     f"xyz={np.round(new_pose_6d[:3]*1000)}mm")

                # Update sensor state
                try:
                    obs_actual = piper.get_observation()
                    last_joints = np.array(
                        [obs_actual[f"joint_{i+1}.pos"] for i in range(6)]
                    )
                except Exception:
                    last_joints = joints_cmd.copy()

            else:
                # === Step 1: interpolate (UMI pattern) ===
                pose_command_6d = pose_interp(t_now)

                # === Step 2: IK + safety + send ===
                ik_total += 1
                try:
                    T_target = np.eye(4)
                    T_target[:3, 3] = pose_command_6d[:3]
                    T_target[:3, :3] = Rotation.from_rotvec(
                        pose_command_6d[3:]
                    ).as_matrix()

                    joints_target = kin.inverse_kinematics(
                        last_joints,
                        T_target,
                        position_weight=self.position_weight,
                        orientation_weight=self.orientation_weight,
                    )
                    joints_safe = np.clip(
                        joints_target,
                        JOINT_LIMITS_DEG["min"] - JOINT_LIMIT_TOLERANCE_DEG,
                        JOINT_LIMITS_DEG["max"] + JOINT_LIMIT_TOLERANCE_DEG,
                    )
                    max_step = self.max_vel_deg_s * dt
                    error = joints_safe - last_joints
                    max_error = np.max(np.abs(error))
                    if max_error > max_step:
                        joints_cmd = last_joints + error * (max_step / max_error)
                    else:
                        joints_cmd = joints_safe

                    action = {f"joint_{i+1}.pos": joints_cmd[i] for i in range(6)}
                    action["gripper.pos"] = grip_value

                    if not self.dry_run:
                        piper.send_action(action)

                    # Read actual sensor state for next IK seed + RingBuffer.
                    sensor_ok = False
                    try:
                        obs_actual = piper.get_observation()
                        last_joints = np.array(
                            [obs_actual[f"joint_{i+1}.pos"] for i in range(6)]
                        )
                        sensor_ok = True
                    except Exception as e:
                        try:
                            status = piper._iface.get_status_deg()
                            last_joints = np.array(
                                [status[f"joint_{i+1}.pos"]
                                 * piper.config.joint_signs[i]
                                 for i in range(6)]
                            )
                            sensor_ok = True
                        except Exception as e2:
                            if sensor_failures < 10:
                                logger.warning(
                                    f"Controller: sensor read failed "
                                    f"(piper: {e}, sdk: {e2})"
                                )
                            last_joints = joints_cmd.copy()

                    if self.verbose and ik_total % 50 == 0:
                        logger.info(
                            f"Controller: tick={ik_total} "
                            f"sensor_ok={sensor_ok} "
                            f"joints={np.round(last_joints, 1)} "
                            f"cmd={np.round(joints_cmd, 1)}"
                        )
                except Exception as e:
                    ik_errors += 1
                    if ik_errors <= 5:
                        logger.warning(f"Controller: IK failed: {e}")

            # === Step 3: update state ring buffer ===
            T_ee = kin.forward_kinematics(last_joints)
            ee_pose_6d = np.concatenate(
                [T_ee[:3, 3], Rotation.from_matrix(T_ee[:3, :3]).as_rotvec()]
            )
            try:
                self.ring_buffer.put(
                    {
                        "ActualJointState": last_joints,
                        "ActualEEPose": ee_pose_6d,
                        "gripper": np.float64(grip_value),
                        "robot_timestamp": np.float64(time.time()),
                        "robot_timestamp_mono": np.float64(time.monotonic()),
                    },
                    wait=False,
                )
            except TimeoutError:
                pass

            # === Step 4: fetch command (UMI: max 1 per cycle) ===
            try:
                command = self.input_queue.get()
                n_cmd = 1
            except Empty:
                n_cmd = 0

            # === Step 5: execute command (UMI pattern) ===
            if n_cmd > 0:
                cmd_val = int(command["cmd"])

                if cmd_val == Command.STOP.value:
                    keep_running = False

                elif cmd_val == Command.SERVOL.value:
                    target_pose = command["target_pose"]
                    duration = float(command["duration"])
                    curr_time = t_now + dt
                    t_insert = curr_time + duration
                    pose_interp = pose_interp.drive_to_waypoint(
                        pose=target_pose,
                        time=t_insert,
                        curr_time=curr_time,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                    )
                    last_waypoint_time = t_insert

                elif cmd_val == Command.SCHEDULE_WAYPOINT.value:
                    target_pose = command["target_pose"]
                    target_time = float(command["target_time"])
                    # translate global time to monotonic time (UMI pattern)
                    target_time = time.monotonic() - time.time() + target_time
                    curr_time = t_now + dt
                    pose_interp = pose_interp.schedule_waypoint(
                        pose=target_pose,
                        time=target_time,
                        max_pos_speed=self.max_pos_speed,
                        max_rot_speed=self.max_rot_speed,
                        curr_time=curr_time,
                        last_waypoint_time=last_waypoint_time,
                    )
                    last_waypoint_time = target_time
                    grip_value = float(command["gripper"])

                elif cmd_val == Command.MOVE_JOINTS.value:
                    joint_move_start = last_joints.copy()
                    joint_move_end = np.asarray(command["target_pose"], dtype=np.float64)
                    duration = float(command["duration"])
                    joint_move_t_start = t_now + dt
                    joint_move_t_end = joint_move_t_start + duration
                    joint_move_gripper = float(command["gripper"])
                    joint_move_active = True
                    if self.verbose:
                        logger.info(
                            f"Controller: MOVE_JOINTS → "
                            f"{np.round(joint_move_end, 1)} over {duration:.1f}s"
                        )

            # === Step 6: regulate frequency (UMI pattern) ===
            t_wait_util = t_start + (iter_idx + 1) * dt
            precise_wait(t_wait_util, time_func=time.monotonic)
            iter_idx += 1

            # Signal ready after first tick
            if iter_idx == 1:
                self.ready_event.set()

            if self.verbose and iter_idx % 100 == 0:
                logger.info(
                    f"Controller: tick={iter_idx} "
                    f"IK_errors={ik_errors}/{ik_total} "
                    f"sensor_fail={sensor_failures}"
                )

        # Cleanup
        try:
            piper.disconnect()
        except Exception:
            pass
        self.ready_event.set()
