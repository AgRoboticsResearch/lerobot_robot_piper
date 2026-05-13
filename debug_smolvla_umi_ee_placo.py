#!/usr/bin/env python
"""Real-time placo meshcat visualization + interactive execution for SmolVLA UMI EE-pose on Piper arm.

Reads real robot joints via CAN, runs SmolVLA inference on camera images,
and visualizes everything in a browser-based 3D viewer:
  - Piper URDF at live joint positions
  - Predicted EE trajectories as colored point clouds
  - Link frame axes (RGB) at each joint

With --execute: interactive human-in-the-loop execution.
  Model keeps inferring and showing predictions in placo.
  Press 'e' to execute the currently displayed action chunk.
  Press 'r' to re-infer, 'q' to quit.

Usage (viz-only, arm can be moved by hand):
  conda activate lerobot_piper_sroi && python lerobot_robot_piper/debug_smolvla_umi_ee_placo.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322274337', width: 640, height: 480, fps: 30} }" \
      --piper can0

Usage (interactive execution):
  conda activate lerobot_piper_sroi && python lerobot_robot_piper/debug_smolvla_umi_ee_placo.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322274337', width: 640, height: 480, fps: 30} }" \
      --piper can0 \
      --execute
"""

import argparse
import atexit
import json
import logging
import os
import select
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robust non-blocking keyboard reader
# ---------------------------------------------------------------------------
class _KeyboardReader:
    """Singleton that sets terminal raw mode once and reads keys without Enter.

    - Sets raw mode on first call to ``start()`` (called automatically).
    - Restores original terminal settings via ``atexit`` — always, even on crash.
    - ``read()`` is non-blocking: returns a key character or ``None``.
    - Drains any buffered keypresses so 'q'/'f'/'s' never queues up stale.
    """

    _instance = None

    def __init__(self):
        self._fd = None
        self._old_term = None
        self._started = False

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self):
        if self._started:
            return
        if not sys.stdin.isatty():
            self._started = True
            return
        try:
            import tty, termios
            self._fd = sys.stdin.fileno()
            self._old_term = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
            atexit.register(self.restore)
            self._started = True
        except Exception:
            self._started = True  # don't retry

    def restore(self):
        if self._old_term is not None and self._fd is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            except Exception:
                pass
            self._old_term = None

    def read(self):
        """Non-blocking read. Returns lowercase char or None."""
        if not self._started:
            self.start()
        if self._fd is None:
            # Non-TTY fallback
            if select.select([sys.stdin], [], [], 0.0)[0]:
                return (sys.stdin.readline().strip().lower() or None)
            return None
        # Drain all buffered characters, return the *last* one
        last = None
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            ch = os.read(self._fd, 1).decode("utf-8", errors="ignore")
            if ch:
                last = ch.lower()
            else:
                break
        return last


_kb = _KeyboardReader.get()


def _check_key():
    """Non-blocking single keypress check. Returns key character or None."""
    return _kb.read()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

URDF_SRC = str(
    PROJECT_ROOT / "lerobot_robot_piper" / "lerobot_robot_piper" / "urdf" / "piper_description.urdf"
)
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_POSE_DEG = [0.0, 61.00, -13.00, 0.00, -36.00, 0.0]
ALL_LINK_NAMES = [
    "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
    "ee_link", "camera_link",
]
JOINT_LIMITS_DEG = {
    "min": [-150.0, 0.0, -170.0, -100.0, -70.0, -120.0],
    "max": [150.0, 180.0, 0.0, 100.0, 70.0, 120.0],
}
JOINT_LIMIT_TOLERANCE_DEG = 0.5
MAX_JOINT_STEP_DEG = 10.0
MAX_IK_POS_ERROR_MM = 10.0


# ---------------------------------------------------------------------------
# Pose conversion utilities (from run_smolvla_inference.py)
# ---------------------------------------------------------------------------

def action_to_pose(action_7d: np.ndarray) -> np.ndarray:
    """Convert 7D action [x, y, z, wx, wy, wz, gripper] to 4x4 pose matrix."""
    T = np.eye(4)
    T[:3, 3] = action_7d[:3]
    rotvec = action_7d[3:6]
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


def pose_to_ee_state(T: np.ndarray, gripper: float = 1.0) -> np.ndarray:
    """Convert 4x4 pose matrix to 7D EE state [x, y, z, wx, wy, wz, gripper]."""
    pos = T[:3, 3]
    R = T[:3, :3]
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-10:
        rotvec = np.zeros(3)
    else:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2 * np.sin(angle))
        rotvec = axis * angle
    return np.array([*pos, *rotvec, gripper], dtype=np.float32)


def compute_ee_delta_state(T_current_world: np.ndarray, T_start_world: np.ndarray,
                           gripper: float = 1.0) -> np.ndarray:
    """Compute EE state in ee_link frame convention relative to starting pose.

    Matches training data convention from transform_trajectory_to_ee.py:
      T_delta = inv(T_start) @ T_current
    Then extracts [x, y, z, wx, wy, wz, gripper] from T_delta.

    This ensures the model receives coordinates in the same frame it was
    trained on (ee_link frame, relative to start), not world frame.
    """
    T_delta = np.linalg.inv(T_start_world) @ T_current_world
    return pose_to_ee_state(T_delta, gripper)


def delta_action_to_world_pose(action_7d: np.ndarray, T_start_world: np.ndarray) -> np.ndarray:
    """Convert a predicted EE action from ee_link-frame delta back to world frame.

    Inverse of compute_ee_delta_state: T_world = T_start @ T_delta
    """
    T_delta = action_to_pose(action_7d[:6])
    T_world = T_start_world @ T_delta
    return T_world


# ---------------------------------------------------------------------------
# Camera config parsing (from run_smolvla_inference.py)
# ---------------------------------------------------------------------------

def parse_cameras_config(cameras_str: str) -> dict:
    """Parse cameras configuration from YAML string."""
    from lerobot.cameras import CameraConfig, make_cameras_from_configs
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras_dict = yaml.safe_load(cameras_str)
    cameras = {}
    for name, config in cameras_dict.items():
        camera_type = config.pop("type")
        if camera_type == "opencv":
            if isinstance(config.get("index_or_path"), int):
                config["index_or_path"] = str(config["index_or_path"])
            cameras[name] = OpenCVCameraConfig(**config)
        elif camera_type == "intelrealsense":
            cameras[name] = RealSenseCameraConfig(**config)
        else:
            cameras[name] = CameraConfig.get_choice_class(camera_type)(**config)
    return cameras


def build_temporal_image(img: np.ndarray, history: list | None = None, horizon: int = 1):
    """Build temporal image observation. For horizon=1, returns (1, C, H, W)."""
    if history is None:
        history = []
    img_float = img.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))
    history.append(img_chw.copy())
    history = history[-horizon:]
    while len(history) < horizon:
        history.insert(0, img_chw.copy())
    return np.stack(history, axis=0), history


# ---------------------------------------------------------------------------
# SmolVLA pipeline loading (from diagnose_predictions.py)
# ---------------------------------------------------------------------------

def load_smolvla_pipeline(pretrained_path: str, device: str = "cuda", dataset_root: str | None = None):
    """Load SmolVLA policy, preprocessor, postprocessor, and task prompt.

    Uses stats already saved in dataset metadata from training — no recompute.
    Returns (policy, preprocessor, postprocessor, task_prompt).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    pretrained_path = str(pretrained_path)

    # Load train_config.json for exact training flags and dataset info
    config_path = Path(pretrained_path) / "train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        policy_config = train_config.get("policy", {})
        ds_cfg = train_config.get("dataset", {})
    else:
        policy_config = {}
        ds_cfg = {}

    # Determine dataset root from training config (CLI override takes priority)
    ds_repo_id = ds_cfg.get("repo_id", "sroi_piper_strawberry_picking")
    if dataset_root:
        ds_root = dataset_root
    else:
        ds_root = ds_cfg.get("root", str(PROJECT_ROOT / "Datasets" / ds_repo_id))

    logger.info(f"Loading stats from training dataset: {ds_repo_id} at {ds_root}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    ds_meta = LeRobotDatasetMetadata(ds_repo_id, root=ds_root)

    # Build SmolVLA config matching training
    cfg = SmolVLAConfig(
        derive_state_from_action=policy_config.get("derive_state_from_action", True),
        use_relative_actions=policy_config.get("use_relative_actions", True),
        relative_exclude_joints=policy_config.get("relative_exclude_joints", ["gripper"]),
        relative_exclude_state_joints=policy_config.get("relative_exclude_state_joints", ["gripper"]),
        pose_dim=policy_config.get("pose_dim", 0),
        device=device,
        resize_imgs_with_padding=tuple(policy_config.get("resize_imgs_with_padding", (512, 512))),
        freeze_vision_encoder=policy_config.get("freeze_vision_encoder", True),
        train_expert_only=policy_config.get("train_expert_only", True),
        train_state_proj=policy_config.get("train_state_proj", True),
        load_vlm_weights=False,
        push_to_hub=False,
        pretrained_path=pretrained_path,
    )

    logger.info(f"Config: derive_state={cfg.derive_state_from_action}, "
                f"relative_actions={cfg.use_relative_actions}")

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=pretrained_path,
        dataset_stats=ds_meta.stats,
    )

    logger.info(f"Preprocessor: {len(preprocessor.steps)} steps")
    for i, step in enumerate(preprocessor.steps):
        logger.info(f"  [{i}] {type(step).__name__}: enabled={getattr(step, 'enabled', 'N/A')}")
    logger.info(f"Postprocessor: {len(postprocessor.steps)} steps")
    for i, step in enumerate(postprocessor.steps):
        logger.info(f"  [{i}] {type(step).__name__}: enabled={getattr(step, 'enabled', 'N/A')}")

    # Resolve task prompt
    task_prompt = None
    if hasattr(ds_meta, "tasks") and len(ds_meta.tasks) > 0:
        task_prompt = ds_meta.tasks.index[0]
    if not task_prompt:
        task_prompt = "perform the task"
    logger.info(f"Task prompt: '{task_prompt}'")

    return policy, preprocessor, postprocessor, task_prompt


# ---------------------------------------------------------------------------
# Placo visualization
# ---------------------------------------------------------------------------

def init_placo_viz():
    """Load Piper URDF and open meshcat viewer.

    Returns dict with robot, viz, and viz helper functions.
    """
    import placo
    from placo_utils.visualization import robot_viz, frame_viz, points_viz

    robot = placo.RobotWrapper(URDF_SRC)
    for jn, val in zip(ARM_JOINTS, HOME_POSE_DEG):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    T_world_ee = robot.get_T_world_frame("ee_link")

    viz = robot_viz(robot)
    viz.display(robot.state.q)
    time.sleep(1.0)  # Give meshcat time to load URDF meshes

    # Show link frames (RGB axes) at each joint
    for frame_name in ALL_LINK_NAMES:
        try:
            T = robot.get_T_world_frame(frame_name)
            frame_viz(frame_name, T)
        except Exception:
            pass

    logger.info(f"placo meshcat viewer: http://127.0.0.1:7001/static/")
    logger.info(f"Initial EE pose: pos=({T_world_ee[0,3]*1000:.0f}, "
                f"{T_world_ee[1,3]*1000:.0f}, {T_world_ee[2,3]*1000:.0f}) mm")

    return {
        "robot": robot,
        "viz": viz,
        "T_world_ee": T_world_ee,
        "frame_viz": frame_viz,
        "points_viz": points_viz,
    }


def update_placo_viz(viz_state: dict, pred_actions: np.ndarray, step_count: int):
    """Update meshcat with predicted action trajectory.

    pred_actions: (50, 7) — already ABSOLUTE after postprocessor.
    Shows colored gradient trajectory, frame markers, and gripper state.
    """
    from placo_utils.visualization import frame_viz as fv, points_viz as pv

    n = len(pred_actions)

    # Convert all timesteps to 3D points
    points = np.zeros((n, 3))
    for t in range(n):
        T_abs = action_to_pose(pred_actions[t])
        points[t] = T_abs[:3, 3]

    # Colored gradient dots only (green→red) — no frame markers on intermediate steps
    n_segments = min(5, n)
    seg_len = max(1, n // n_segments)
    colors = [0x00FF00, 0x88FF00, 0xFFFF00, 0xFF8800, 0xFF0000]
    for s in range(n_segments):
        start = s * seg_len
        end = min((s + 1) * seg_len, n)
        if start < n:
            seg_points = points[start:end]
            if len(seg_points) > 0:
                pv(f"pred_seg{s}", seg_points, radius=0.004, color=colors[min(s, len(colors) - 1)])

    # RGB frame marker only on the last predicted point
    T_last = action_to_pose(pred_actions[-1])
    fv("pred_end", T_last)


def update_robot_viz(viz_state: dict, joint_deg: np.ndarray):
    """Update placo robot visualization with current joint positions."""
    robot = viz_state["robot"]
    viz = viz_state["viz"]
    frame_viz = viz_state["frame_viz"]

    for jn, val in zip(ARM_JOINTS, joint_deg):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()
    viz.display(robot.state.q)

    # Update link frames to follow real joints
    for f in ALL_LINK_NAMES:
        try:
            T = robot.get_T_world_frame(f)
            frame_viz(f, T)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Execution functions (from run_smolvla_inference.py)
# ---------------------------------------------------------------------------

def validate_joint_positions(joints_deg, prev_joints_deg, step_label=""):
    """Safety-check IK output before sending to robot."""
    label = f" [{step_label}]" if step_label else ""
    if np.any(np.isnan(joints_deg)) or np.any(np.isinf(joints_deg)):
        raise RuntimeError(f"NaN/Inf in IK solution{label}: {joints_deg}")
    for j in range(len(joints_deg)):
        lo = JOINT_LIMITS_DEG["min"][j]
        hi = JOINT_LIMITS_DEG["max"][j]
        if joints_deg[j] < lo - JOINT_LIMIT_TOLERANCE_DEG or joints_deg[j] > hi + JOINT_LIMIT_TOLERANCE_DEG:
            raise RuntimeError(f"Joint {j+1}={joints_deg[j]:.1f}° outside limit [{lo}, {hi}]°{label}")
    delta = joints_deg - prev_joints_deg
    max_d = np.max(np.abs(delta))
    if max_d > MAX_JOINT_STEP_DEG:
        delta = delta * (MAX_JOINT_STEP_DEG / max_d)
        joints_deg = prev_joints_deg + delta
        logger.debug(f"Step cap applied{label}: {max_d:.1f}° → {MAX_JOINT_STEP_DEG}°")
    return joints_deg


def emergency_hold(piper_robot, joint_deg, gripper_val):
    """Emergency stop: hold robot at current position."""
    for _ in range(3):
        action = {f"joint_{i+1}.pos": joint_deg[i] for i in range(6)}
        action["gripper.pos"] = gripper_val
        piper_robot.send_action(action)
        time.sleep(0.05)


def execute_action_steps(
    pred_np, piper_robot, kin,
    current_joint_deg, current_gripper,
    n_execute_steps=8, dt=0.05,
    position_weight=1.0, orientation_weight=0.01,
):
    """Execute N steps of predicted action chunk on the real robot.

    pred_np: (50, 7) absolute EE poses [x,y,z,rx,ry,rz,gripper]
    Returns (final_joint_deg, final_gripper).
    """
    n = min(n_execute_steps, len(pred_np))
    obs = piper_robot.get_observation()
    prev_joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
    prev_gripper = obs.get("gripper.pos", current_gripper)

    T_current = kin.forward_kinematics(prev_joints)
    logger.info(f"  Current EE: pos=[{T_current[0,3]:.4f},{T_current[1,3]:.4f},{T_current[2,3]:.4f}] "
                f"joints=[{prev_joints[0]:+.1f},{prev_joints[1]:+.1f},{prev_joints[2]:+.1f},"
                f"{prev_joints[3]:+.1f},{prev_joints[4]:+.1f},{prev_joints[5]:+.1f}]")

    for t in range(n):
        action_7d = pred_np[t]
        T_target = action_to_pose(action_7d[:6])
        gripper_val = float(action_7d[6])

        try:
            j_ik = kin.inverse_kinematics(
                prev_joints, T_target,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
        except Exception as e:
            logger.error(f"  IK failed step {t}: {e}")
            continue

        T_check = kin.forward_kinematics(j_ik)
        err_mm = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]) * 1000
        if err_mm > MAX_IK_POS_ERROR_MM:
            if t == 0:
                logger.warning(f"  IK error {err_mm:.1f}mm at step {t}\n"
                               f"    target pos: [{T_target[0,3]:.4f},{T_target[1,3]:.4f},{T_target[2,3]:.4f}]\n"
                               f"    IK result:  [{T_check[0,3]:.4f},{T_check[1,3]:.4f},{T_check[2,3]:.4f}]")
            else:
                logger.warning(f"  IK error {err_mm:.1f}mm at step {t}, skipping")
            continue

        try:
            j_safe = validate_joint_positions(j_ik, prev_joints, step_label=f"step{t}")
        except RuntimeError as e:
            logger.error(f"  Safety reject step {t}: {e}")
            continue

        action = {f"joint_{i+1}.pos": j_safe[i] for i in range(6)}
        action["gripper.pos"] = gripper_val
        piper_robot.send_action(action)
        time.sleep(dt)
        prev_joints = j_safe.copy()
        prev_gripper = gripper_val

        if (t + 1) % 4 == 0 or t == n - 1:
            logger.info(f"  executed step {t+1}/{n} | err={err_mm:.2f}mm | "
                        f"J=[{j_safe[0]:+.1f},{j_safe[1]:+.1f},{j_safe[2]:+.1f},"
                        f"{j_safe[3]:+.1f},{j_safe[4]:+.1f},{j_safe[5]:+.1f}] | "
                        f"grip={gripper_val:.3f}")

    return prev_joints, prev_gripper


def execute_auto_steps(
    pred_np, piper_robot, kin,
    current_joint_deg, current_gripper,
    n_execute_steps=8, dt=0.05,
    position_weight=1.0, orientation_weight=0.01,
):
    """Execute N steps of predicted action chunk with non-blocking freeze check.

    Same as execute_action_steps but checks for 'f' keypress between each step.
    Returns (final_joint_deg, final_gripper, frozen: bool).
    """
    n = min(n_execute_steps, len(pred_np))
    obs = piper_robot.get_observation()
    prev_joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
    prev_gripper = obs.get("gripper.pos", current_gripper)

    T_current = kin.forward_kinematics(prev_joints)
    T_first_target = action_to_pose(pred_np[0][:6])
    logger.info(f"  Current EE (FK): pos=[{T_current[0,3]*1000:.1f},{T_current[1,3]*1000:.1f},{T_current[2,3]*1000:.1f}]mm")
    logger.info(f"  Step 0 target:   pos=[{T_first_target[0,3]*1000:.1f},{T_first_target[1,3]*1000:.1f},{T_first_target[2,3]*1000:.1f}]mm")

    frozen = False
    for t in range(n):
        # Non-blocking keyboard check for freeze toggle
        key = _check_key()
        if key == "f":
            logger.warning("FREEZE requested — holding position")
            emergency_hold(piper_robot, prev_joints, prev_gripper)
            frozen = True
            break
        elif key == "q":
            logger.info("Quit requested during auto execution")
            frozen = True
            break

        action_7d = pred_np[t]
        T_target = action_to_pose(action_7d[:6])
        gripper_val = float(action_7d[6])

        try:
            j_ik = kin.inverse_kinematics(
                prev_joints, T_target,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
        except Exception as e:
            logger.error(f"  IK failed step {t}: {e}")
            continue

        T_check = kin.forward_kinematics(j_ik)
        err_mm = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]) * 1000
        if err_mm > MAX_IK_POS_ERROR_MM:
            if t <= 1:
                logger.warning(f"  IK error {err_mm:.1f}mm at step {t}")
                logger.warning(f"    target  pos: {T_target[:3,3]*1000} mm")
                logger.warning(f"    IK gave pos: {T_check[:3,3]*1000} mm")
                logger.warning(f"    prev joints: {prev_joints}")
                logger.warning(f"    IK  joints:  {j_ik}")
            else:
                logger.debug(f"  IK error {err_mm:.1f}mm at step {t}, skipping")
            continue

        try:
            j_safe = validate_joint_positions(j_ik, prev_joints, step_label=f"step{t}")
        except RuntimeError as e:
            logger.error(f"  Safety reject step {t}: {e}")
            continue

        action = {f"joint_{i+1}.pos": j_safe[i] for i in range(6)}
        action["gripper.pos"] = gripper_val
        piper_robot.send_action(action)
        time.sleep(dt)
        prev_joints = j_safe.copy()
        prev_gripper = gripper_val

        if (t + 1) % 4 == 0 or t == n - 1:
            logger.info(f"  auto step {t+1}/{n} | err={err_mm:.2f}mm | "
                        f"J=[{j_safe[0]:+.1f},{j_safe[1]:+.1f},{j_safe[2]:+.1f},"
                        f"{j_safe[3]:+.1f},{j_safe[4]:+.1f},{j_safe[5]:+.1f}] | "
                        f"grip={gripper_val:.3f}")

    return prev_joints, prev_gripper, frozen


# ---------------------------------------------------------------------------
# Main real-time visualization loop
# ---------------------------------------------------------------------------

def run_realtime_viz(args):
    """Real-time inference loop with placo meshcat visualization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # --- Load SmolVLA pipeline ---
    policy, preprocessor, postprocessor, task_prompt = load_smolvla_pipeline(
        args.pretrained_path, device, dataset_root=getattr(args, "dataset_root", None)
    )

    # Override task if provided
    if args.task:
        task_prompt = args.task

    # --- Init placo visualization ---
    placo_viz = init_placo_viz()

    # --- Connect to real Piper ---
    from lerobot_robot_piper.piper import Piper
    from lerobot_robot_piper.config_piper import PiperConfig

    logger.info(f"Connecting to Piper on {args.piper}...")
    config = PiperConfig(
        can_interface=args.piper,
        include_gripper=True,
        use_degrees=True,
        cameras={},
    )
    piper_robot = Piper(config)
    piper_robot.connect()
    logger.info("Piper connected and enabled (arm + gripper)")

    # --- IK solver for execution/auto mode ---
    kin = None
    if args.execute or args.auto:
        from lerobot.model.kinematics import RobotKinematics
        kin = RobotKinematics(
            urdf_path=URDF_SRC,
            target_frame_name="ee_link",
            joint_names=ARM_JOINTS,
        )
        logger.info("IK solver initialized for execution/auto mode")

        # Move to home pose before starting
        logger.info("Moving to home pose...")
        home = np.array(HOME_POSE_DEG, dtype=np.float64)
        obs = piper_robot.get_observation()
        current_joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
        n_interp = 50
        for step in range(n_interp + 1):
            alpha = step / n_interp
            target = current_joints * (1 - alpha) + home * alpha
            action = {f"joint_{i+1}.pos": target[i] for i in range(6)}
            action["gripper.pos"] = 1.0
            piper_robot.send_action(action)
            time.sleep(0.05)
        logger.info("Home pose reached")

    # --- Record starting EE pose (reference frame for ee_link-frame deltas) ---
    robot = placo_viz["robot"]
    home_joints = np.array(HOME_POSE_DEG, dtype=np.float64)
    for jn, val in zip(ARM_JOINTS, home_joints):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()
    T_start = robot.get_T_world_frame("ee_link").copy()
    logger.info(f"Reference EE pose (T_start): pos=({T_start[0,3]*1000:.1f}, "
                f"{T_start[1,3]*1000:.1f}, {T_start[2,3]*1000:.1f}) mm")

    # --- Motor mode ---
    # In execute/auto mode: keep motors enabled
    # In viz-only mode: disable torque so arm can be moved by hand
    if not args.execute and not args.auto:
        logger.info("Disabling motor torque for read-only mode...")
        try:
            piper_robot._iface.piper.DisablePiper()
            logger.info("Motors disabled — you can now move the arm freely by hand")
        except Exception as e:
            logger.warning(f"Could not disable motors: {e}")

    # --- Connect cameras ---
    from lerobot.cameras import make_cameras_from_configs

    cameras_config = parse_cameras_config(args.cameras)
    if not cameras_config:
        raise ValueError("No cameras configured")

    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Camera connected: {cam_name}")

    # --- Reset pipeline state ---
    policy.reset()
    for step in preprocessor.steps:
        if hasattr(step, "reset"):
            step.reset()

    # --- State ---
    image_history = {}
    _has_gui = True
    _frozen = False
    step_count = 0

    logger.info(f"\n{'='*60}")
    logger.info("  Real-time SmolVLA visualization started")
    logger.info("  Open http://127.0.0.1:7001/static/ in your browser")
    if args.auto:
        logger.info(f"  AUTO MODE — executing {args.n_execute_steps} steps per cycle")
        logger.info("  Press 's' to START, 'f' to freeze/unfreeze, 'q' to quit")
    elif args.execute:
        logger.info(f"  EXECUTION MODE — press 'e' to execute {args.n_execute_steps} steps, 'q' to quit")
        logger.info("  Ctrl+C = emergency stop (hold position)")
    else:
        logger.info("  Press Ctrl+C to stop")
    logger.info(f"{'='*60}\n")

    # --- Initialize keyboard reader ---
    _kb.start()

    # --- Auto mode: wait for 's' to start ---
    if args.auto:
        logger.info("Robot is at home pose. Press 's' to START auto execution, 'q' to quit.")
        while True:
            try:
                obs_wait = piper_robot.get_observation()
                jwait = np.array([obs_wait[f"joint_{i+1}.pos"] for i in range(6)])
                update_robot_viz(placo_viz, jwait)
            except Exception:
                pass
            key = _check_key()
            if key == "s":
                logger.info("Auto execution STARTED")
                break
            elif key == "q":
                logger.info("Quit before starting")
                return
            time.sleep(0.05)

    try:
        last_log_time = 0.0
        while args.max_steps == 0 or step_count < args.max_steps:
            t0 = time.perf_counter()

            # --- Read Piper state ---
            try:
                obs = piper_robot.get_observation()
                joint_deg = np.array([
                    obs["joint_1.pos"], obs["joint_2.pos"],
                    obs["joint_3.pos"], obs["joint_4.pos"],
                    obs["joint_5.pos"], obs["joint_6.pos"],
                ])
                gripper_val = obs.get("gripper.pos", 1.0)
            except Exception as e:
                logger.error(f"Failed to read Piper state: {e}")
                time.sleep(0.1)
                continue

            # FK to get EE pose
            robot = placo_viz["robot"]
            for jn, val in zip(ARM_JOINTS, joint_deg):
                robot.set_joint(jn, np.deg2rad(val))
            robot.update_kinematics()
            T_ee_world = robot.get_T_world_frame("ee_link")

            # Convert to ee_link-frame delta from T_start (matches training convention)
            obs_state = compute_ee_delta_state(T_ee_world, T_start, gripper_val)

            # --- Update placo with real robot state ---
            update_robot_viz(placo_viz, joint_deg)

            # --- Build batch ---
            batch = {
                "observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device),
                "task": [task_prompt],
            }

            # Read cameras
            camera_images = {}
            for cam_name, camera in cameras.items():
                img = camera.read()
                camera_images[cam_name] = img

                temporal_img, hist_list = build_temporal_image(
                    img,
                    history=image_history.get(cam_name),
                    horizon=1,
                )
                image_history[cam_name] = hist_list

                batch[f"observation.images.{cam_name}"] = (
                    torch.from_numpy(temporal_img).unsqueeze(0).to(device)
                )

            # --- Inference ---
            t_infer = time.perf_counter()
            with torch.no_grad():
                processed = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed)
                pred_abs = postprocessor(pred_actions)

            pred_np = pred_abs[0].cpu().numpy()  # (50, 7) — ee_link-frame deltas
            infer_ms = (time.perf_counter() - t_infer) * 1000

            # Convert predictions from ee_link-frame delta to world frame for viz/IK
            pred_world = np.zeros_like(pred_np)
            for t in range(len(pred_np)):
                T_world = delta_action_to_world_pose(pred_np[t], T_start)
                pos = T_world[:3, 3]
                R = T_world[:3, :3]
                angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                if angle < 1e-10:
                    rotvec = np.zeros(3)
                else:
                    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
                    axis = axis / (2 * np.sin(angle))
                    rotvec = axis * angle
                pred_world[t] = [*pos, *rotvec, pred_np[t, 6]]

            # --- Update placo with predictions (world frame) ---
            update_placo_viz(placo_viz, pred_world, step_count)

            # --- Print detailed log every 1s ---
            now = time.perf_counter()
            if now - last_log_time >= 1.0 or step_count < 3:
                last_log_time = now
                # Model input
                proc_state = processed.get("observation.state", None)
                print(f"\n--- step {step_count} | infer {infer_ms:.0f}ms ---")
                if proc_state is not None:
                    ps = proc_state[0].cpu().numpy()
                    print(f"  model input state ({ps.shape[0]}):")
                    print(f"    [{','.join(f'{v:.6f}' for v in ps)}]")
                else:
                    print(f"  model input state: N/A")
                print(f"  joints: [{joint_deg[0]:+.1f},{joint_deg[1]:+.1f},{joint_deg[2]:+.1f},"
                      f"{joint_deg[3]:+.1f},{joint_deg[4]:+.1f},{joint_deg[5]:+.1f}]")
                # Predicted action chunk summary (world frame)
                print(f"  pred chunk ({pred_world.shape[0]} steps, world frame):")
                # Print first, every 10th, and last step
                indices = sorted(set([0] + list(range(9, pred_world.shape[0], 10)) + [pred_world.shape[0] - 1]))
                for i in indices:
                    a = pred_world[i]
                    print(f"    [{i:2d}] xyz=[{a[0]:.4f},{a[1]:.4f},{a[2]:.4f}] "
                          f"rot=[{a[3]:.4f},{a[4]:.4f},{a[5]:.4f}] grip={a[6]:.3f}")
                print()

            # --- Show camera feed if requested ---
            if args.cameraview and _has_gui:
                for cam_name, img in camera_images.items():
                    display = img.copy()
                    if display.ndim == 3 and display.shape[2] == 3:
                        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                    try:
                        cv2.imshow(f"Camera: {cam_name}", display)
                    except cv2.error:
                        _has_gui = False
                        logger.warning("OpenCV headless — disabling cameraview")
                        break
                if _has_gui:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break

            step_count += 1

            # --- Interactive execution mode ---
            if args.execute and kin is not None:
                logger.info(f"Press 'e' to execute {args.n_execute_steps} steps, 'r' to re-infer, 'q' to quit")

                # Blocking wait for keypress — keeps updating viz
                cmd = None
                while cmd is None:
                    try:
                        obs_live = piper_robot.get_observation()
                        jlive = np.array([obs_live[f"joint_{i+1}.pos"] for i in range(6)])
                        update_robot_viz(placo_viz, jlive)
                    except Exception:
                        pass
                    cmd = _check_key()
                    if cmd is None:
                        time.sleep(0.05)

                if cmd == "e":
                    try:
                        logger.info(f"Executing {args.n_execute_steps} steps...")
                        final_joints, final_gripper = execute_action_steps(
                            pred_np=pred_world,
                            piper_robot=piper_robot,
                            kin=kin,
                            current_joint_deg=joint_deg,
                            current_gripper=gripper_val,
                            n_execute_steps=args.n_execute_steps,
                            dt=1.0 / args.fps,
                            position_weight=args.position_weight,
                            orientation_weight=args.orientation_weight,
                        )
                        logger.info("Execution complete. Re-inferring...")
                    except KeyboardInterrupt:
                        logger.warning("EMERGENCY STOP during execution")
                        emergency_hold(piper_robot, joint_deg, gripper_val)
                        raise
                    except Exception as e:
                        logger.error(f"Execution error: {e}")
                        emergency_hold(piper_robot, joint_deg, gripper_val)
                elif cmd == "q":
                    print()
                    break
                elif cmd == "r":
                    logger.info("Re-inferring...")
                else:
                    logger.info(f"Unknown key '{cmd}'. Use e/r/q.")

            # --- Auto execution mode ---
            elif args.auto and kin is not None:

                # Non-blocking single-key check between inference cycles
                key = _check_key()
                if key == "f":
                    _frozen = not _frozen
                    if _frozen:
                        logger.warning("*** FROZEN — press 'f' to unfreeze ***")
                        try:
                            obs_now = piper_robot.get_observation()
                            hold_j = np.array([obs_now[f"joint_{i+1}.pos"] for i in range(6)])
                            hold_g = obs_now.get("gripper.pos", gripper_val)
                            emergency_hold(piper_robot, hold_j, hold_g)
                        except Exception:
                            pass
                    else:
                        logger.info("*** UNFROZEN — re-inferring before resume ***")
                elif key == "q":
                    break

                if _frozen:
                    # Frozen: skip execution, just update viz with live robot state
                    try:
                        obs_live = piper_robot.get_observation()
                        jlive = np.array([obs_live[f"joint_{i+1}.pos"] for i in range(6)])
                        update_robot_viz(placo_viz, jlive)
                    except Exception:
                        pass
                    time.sleep(0.05)
                    continue

                # Execute predicted steps
                try:
                    logger.info(f"AUTO: executing {args.n_execute_steps} steps...")
                    final_joints, final_gripper, frozen = execute_auto_steps(
                        pred_np=pred_world,
                        piper_robot=piper_robot,
                        kin=kin,
                        current_joint_deg=joint_deg,
                        current_gripper=gripper_val,
                        n_execute_steps=args.n_execute_steps,
                        dt=1.0 / args.fps,
                        position_weight=args.position_weight,
                        orientation_weight=args.orientation_weight,
                    )
                    if frozen:
                        _frozen = True
                    else:
                        logger.info("AUTO: execution complete, re-inferring...")
                except Exception as e:
                    logger.error(f"AUTO execution error: {e}")
                    emergency_hold(piper_robot, joint_deg, gripper_val)

            else:
                # Viz-only mode: maintain FPS
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, 1.0 / args.fps - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if piper_robot and (args.execute or args.auto):
            try:
                obs = piper_robot.get_observation()
                hold_joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
                hold_gripper = obs.get("gripper.pos", 0.5)
                emergency_hold(piper_robot, hold_joints, hold_gripper)
                logger.info("Emergency hold applied")
            except Exception:
                pass
    finally:
        _kb.restore()
        if args.cameraview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        for camera in cameras.values():
            camera.disconnect()
        try:
            piper_robot.disconnect()
            logger.info("Piper disconnected")
        except Exception:
            pass
        logger.info(f"Done. Ran {step_count} steps.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time placo visualization for SmolVLA UMI EE-pose on Piper arm"
    )
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to trained SmolVLA checkpoint directory")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Override dataset root path (skip train_config.json dataset.root)")
    parser.add_argument("--cameras", type=str, required=True,
                        help="Camera YAML config (e.g. \"{ color: {type: intelrealsense, ...} }\")")
    parser.add_argument("--piper", type=str, required=True,
                        help="CAN interface for real Piper (e.g. can0)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task prompt (auto-detected from dataset if not provided)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Control loop frequency (default: 30)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference steps (0=infinite)")
    parser.add_argument("--cameraview", action="store_true",
                        help="Show camera feed window")
    parser.add_argument("--execute", action="store_true",
                        help="Enable interactive execution mode (press 'e' to execute)")
    parser.add_argument("--auto", action="store_true",
                        help="Enable auto execution mode (robot executes continuously, 'f' to freeze)")
    parser.add_argument("--n_execute_steps", type=int, default=8,
                        help="Number of action steps to execute per cycle (default: 8)")
    parser.add_argument("--position_weight", type=float, default=1.0,
                        help="IK position weight (default: 1.0)")
    parser.add_argument("--orientation_weight", type=float, default=0.01,
                        help="IK orientation weight (default: 0.01)")
    args = parser.parse_args()

    if args.execute and args.auto:
        parser.error("--execute and --auto are mutually exclusive")

    run_realtime_viz(args)


if __name__ == "__main__":
    main()
