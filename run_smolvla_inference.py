#!/usr/bin/env python
"""Live inference for SmolVLA UMI EE-pose on Piper arm.

Runs the FULL pipeline in real-time: camera → preprocess → model → postprocess → action.

Two modes:
  1. Camera mode (--cameras): Live camera feed, real-time inference loop
  2. Dataset mode (--dataset_root): Replay dataset frames through the pipeline

Optional visualization:
  --placo_viz: Show predicted EE actions as trajectory on Piper URDF via meshcat.
               Requires placo + placo_utils (available in conda env lerobot_piper_sroi).
               Open http://127.0.0.1:7001/static/ in a browser after launch.

Usage (camera mode — requires real robot):
  conda activate lerobot_piper_sroi && python lerobot_robot_piper/run_smolvla_inference.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322274337', width: 640, height: 480, fps: 30} }" \
      --piper can0 \
      --execute \
      --placo_viz

Usage (camera mode — inference only, no execution):
  conda activate lerobot_piper_sroi && python lerobot_robot_piper/run_smolvla_inference.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322274337', width: 640, height: 480, fps: 30} }" \
      --piper can0 \
      --placo_viz

Usage (dataset mode — dry run, no robot needed):
  cd /home/hls/codes/lerobot_piper_sroi && uv run --directory lerobot python lerobot_robot_piper/run_smolvla_inference.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --dataset_root Datasets/sroi_piper_strawberry_picking \
      --max_steps 100
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FPS = 30
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Ensure project root is on sys.path so package imports work from any CWD
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
URDF_SRC = str(PROJECT_ROOT / "lerobot_robot_piper" / "lerobot_robot_piper" / "urdf" / "piper_description.urdf")
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_POSE_DEG = [0.0, 40.11, -45.84, 0.0, 17.19, 0.0]
JOINT_LIMITS_DEG = {
    "min": [-150.0, 0.0, -170.0, -100.0, -70.0, -120.0],
    "max": [150.0, 180.0, 0.0, 100.0, 70.0, 120.0],
}
JOINT_LIMIT_TOLERANCE_DEG = 0.5
MAX_JOINT_STEP_DEG = 10.0
MAX_IK_POS_ERROR_MM = 10.0
ALL_LINK_NAMES = [
    "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
    "ee_link", "camera_link",
]


def parse_cameras_config(cameras_str: str) -> dict[str, CameraConfig]:
    """Parse cameras configuration from YAML string."""
    cameras_dict = yaml.safe_load(cameras_str)
    cameras: dict[str, CameraConfig] = {}
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
    img_chw = np.transpose(img_float, (2, 0, 1))  # (C, H, W)
    history.append(img_chw.copy())
    history = history[-horizon:]

    while len(history) < horizon:
        history.insert(0, img_chw.copy())

    return np.stack(history, axis=0), history  # (T, C, H, W)


def load_smolvla_pipeline(pretrained_path: str, device: str = "cuda", ds_meta=None):
    """Load SmolVLA policy, preprocessor, and postprocessor.

    This MUST match the training config EXACTLY, otherwise normalization
    will be inconsistent and predictions garbage.
    """
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    pretrained_path = str(pretrained_path)

    # Load config from checkpoint
    import json
    config_path = Path(pretrained_path) / "train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        policy_config = train_config.get("policy", {})
        logger.info(f"Loaded training config from {config_path}")
    else:
        policy_config = {}

    # Build SmolVLA config — key UMI flags
    cfg = SmolVLAConfig(
        derive_state_from_action=policy_config.get("derive_state_from_action", True),
        use_relative_actions=policy_config.get("use_relative_actions", True),
        relative_exclude_joints=policy_config.get("relative_exclude_joints", ["gripper"]),
        relative_exclude_state_joints=policy_config.get("relative_exclude_state_joints", ["gripper"]),
        device=device,
        resize_imgs_with_padding=policy_config.get("resize_imgs_with_padding", (512, 512)),
        freeze_vision_encoder=policy_config.get("freeze_vision_encoder", True),
        train_expert_only=policy_config.get("train_expert_only", True),
        train_state_proj=policy_config.get("train_state_proj", True),
        load_vlm_weights=False,
        push_to_hub=False,
        pretrained_path=pretrained_path,
    )

    logger.info(f"Config: derive_state_from_action={cfg.derive_state_from_action}")
    logger.info(f"Config: use_relative_actions={cfg.use_relative_actions}")
    logger.info(f"Config: device={cfg.device}")

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=pretrained_path,
        dataset_stats=ds_meta.stats if ds_meta else None,
    )

    logger.info(f"Preprocessor: {len(preprocessor.steps)} steps")
    for j, step in enumerate(preprocessor.steps):
        logger.info(f"  [{j}] {type(step).__name__}: enabled={getattr(step, 'enabled', 'N/A')}")
    logger.info(f"Postprocessor: {len(postprocessor.steps)} steps")
    for j, step in enumerate(postprocessor.steps):
        logger.info(f"  [{j}] {type(step).__name__}: enabled={getattr(step, 'enabled', 'N/A')}")

    return policy, preprocessor, postprocessor


def action_to_pose(action_7d: np.ndarray) -> np.ndarray:
    """Convert 7D action [x, y, z, wx, wy, wz, gripper] to 4x4 pose matrix."""
    T = np.eye(4)
    T[:3, 3] = action_7d[:3]
    rotvec = action_7d[3:6]
    angle = np.linalg.norm(rotvec)
    if angle > 1e-10:
        axis = rotvec / angle
        c = np.cos(angle)
        s = np.sin(angle)
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


def validate_joint_positions(
    joints_deg: np.ndarray,
    prev_joints_deg: np.ndarray,
    step_label: str = "",
) -> np.ndarray:
    """Safety-check IK output before sending to robot.

    Checks NaN/Inf, joint limits, and caps per-step movement.
    Returns validated (possibly clamped) joint positions.
    Raises RuntimeError on fatal violations.
    """
    label = f" [{step_label}]" if step_label else ""

    # NaN / Inf
    if np.any(np.isnan(joints_deg)) or np.any(np.isinf(joints_deg)):
        raise RuntimeError(f"NaN/Inf in IK solution{label}: {joints_deg}")

    # Joint hard limits
    for j in range(len(joints_deg)):
        lo = JOINT_LIMITS_DEG["min"][j]
        hi = JOINT_LIMITS_DEG["max"][j]
        if joints_deg[j] < lo - JOINT_LIMIT_TOLERANCE_DEG or joints_deg[j] > hi + JOINT_LIMIT_TOLERANCE_DEG:
            raise RuntimeError(
                f"Joint {j+1}={joints_deg[j]:.1f}° outside limit "
                f"[{lo}, {hi}]°{label}"
            )

    # Max step cap
    delta = joints_deg - prev_joints_deg
    max_d = np.max(np.abs(delta))
    if max_d > MAX_JOINT_STEP_DEG:
        delta = delta * (MAX_JOINT_STEP_DEG / max_d)
        joints_deg = prev_joints_deg + delta
        logger.debug(f"Step cap applied{label}: {max_d:.1f}° → {MAX_JOINT_STEP_DEG}°")

    return joints_deg


def emergency_hold(piper_robot, joint_deg: np.ndarray, gripper_val: float):
    """Emergency stop: hold robot at current position."""
    for _ in range(3):
        action = {f"joint_{i+1}.pos": joint_deg[i] for i in range(6)}
        action["gripper.pos"] = gripper_val
        piper_robot.send_action(action)
        time.sleep(0.05)


def execute_action_steps(
    pred_np: np.ndarray,
    piper_robot,
    kin,
    current_joint_deg: np.ndarray,
    current_gripper: float,
    n_execute_steps: int = 8,
    dt: float = 0.05,
    position_weight: float = 1.0,
    orientation_weight: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Execute N steps of the predicted action chunk on the real robot.

    pred_np: (50, 7) absolute EE poses [x,y,z,rx,ry,rz,gripper]
    Returns (final_joint_deg, final_gripper).
    """
    n = min(n_execute_steps, len(pred_np))
    obs = piper_robot.get_observation()
    prev_joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
    prev_gripper = obs.get("gripper.pos", current_gripper)

    # Log current robot pose for debugging
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

        # FK error check
        T_check = kin.forward_kinematics(j_ik)
        err_mm = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]) * 1000
        if err_mm > MAX_IK_POS_ERROR_MM:
            if t == 0:
                # Log detailed info on first failure to help diagnose
                logger.warning(
                    f"  IK error {err_mm:.1f}mm at step {t}\n"
                    f"    target pos: [{T_target[0,3]:.4f},{T_target[1,3]:.4f},{T_target[2,3]:.4f}]\n"
                    f"    IK result:  [{T_check[0,3]:.4f},{T_check[1,3]:.4f},{T_check[2,3]:.4f}]\n"
                    f"    current pos:[{T_current[0,3]:.4f},{T_current[1,3]:.4f},{T_current[2,3]:.4f}]\n"
                    f"    action_7d:  [{action_7d[0]:.4f},{action_7d[1]:.4f},{action_7d[2]:.4f},"
                    f"{action_7d[3]:.4f},{action_7d[4]:.4f},{action_7d[5]:.4f},{action_7d[6]:.3f}]"
                )
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


def init_placo_viz() -> dict:
    """Load Piper URDF and open meshcat viewer.

    Returns dict with robot, viz, T_world_ee.
    """
    import placo
    from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, points_viz

    robot = placo.RobotWrapper(URDF_SRC)
    for jn, val in zip(ARM_JOINTS, HOME_POSE_DEG):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    T_world_ee = robot.get_T_world_frame("ee_link")

    viz = robot_viz(robot)
    viz.display(robot.state.q)

    # Show link frames
    for frame_name in ALL_LINK_NAMES:
        try:
            robot_frame_viz(robot, frame_name)
        except Exception:
            pass
    frame_viz("ee_link", T_world_ee)

    logger.info(f"placo meshcat viewer: http://127.0.0.1:7001/static/")
    logger.info(f"Initial EE pose: pos=({T_world_ee[0,3]*1000:.0f}, {T_world_ee[1,3]*1000:.0f}, {T_world_ee[2,3]*1000:.0f}) mm")

    return {
        "robot": robot,
        "viz": viz,
        "T_world_ee": T_world_ee,
        "frame_viz": frame_viz,
        "points_viz": points_viz,
    }


def update_placo_viz(viz_state: dict, pred_actions: np.ndarray, step_count: int):
    """Update meshcat with predicted action trajectory.

    pred_actions: (50, 7) — 50-timestep action chunk, already ABSOLUTE after
    postprocessor (AbsoluteActionsProcessorStep added back cached obs_state).
    Use action_to_pose() directly — do NOT compose with T_world_ee.
    """
    from placo_utils.visualization import frame_viz as fv, points_viz as pv

    # Convert all 50 timesteps to 3D points (already absolute)
    points = []
    for t in range(len(pred_actions)):
        T_abs = action_to_pose(pred_actions[t])
        points.append(T_abs[:3, 3].copy())
    points = np.array(points)

    # Current (t=0) target pose
    T_world_current = action_to_pose(pred_actions[0])

    # Update meshcat (same-name objects get replaced)
    pv("pred_trajectory", points, radius=0.003, color=0x00AAFF)  # blue
    fv("pred_current", T_world_current)


def run_camera_mode(args):
    """Real-time inference with live cameras."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Initialize placo visualization if requested
    placo_viz = None
    if args.placo_viz:
        try:
            placo_viz = init_placo_viz()
        except ImportError as e:
            logger.warning(f"placo/placo_utils not available: {e}")
            logger.warning("Install in conda env: conda activate lerobot_piper_sroi")
            logger.warning("Continuing without visualization.")

    # Connect to real Piper if requested
    piper_robot = None
    if args.piper:
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

    # Standalone placo robot for FK when piper is connected but placo_viz is off
    fk_robot = None
    if piper_robot and not placo_viz:
        import placo
        fk_robot = placo.RobotWrapper(URDF_SRC)
        logger.info("Standalone FK robot initialized (ee_link frame)")

    # Initialize IK solver for execution mode
    kin = None
    if args.execute and piper_robot:
        from lerobot.model.kinematics import RobotKinematics
        kin = RobotKinematics(
            urdf_path=URDF_SRC,
            target_frame_name="ee_link",
            joint_names=ARM_JOINTS,
        )
        logger.info("RobotKinematics IK solver initialized")

    # Camera mode: load ds_meta from the training dataset for stats and task prompt
    import json
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    config_path = Path(args.pretrained_path) / "train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        ds_cfg = train_config.get("dataset", {})
        ds_repo_id = ds_cfg.get("repo_id", "test_ee_dataset")
        ds_root = ds_cfg.get("root", str(PROJECT_ROOT / "Datasets" / ds_repo_id))
    else:
        ds_repo_id = "test_ee_dataset"
        ds_root = str(PROJECT_ROOT / "Datasets" / ds_repo_id)
    logger.info(f"Loading stats from training dataset: {ds_repo_id} at {ds_root}")
    ds_meta = LeRobotDatasetMetadata(ds_repo_id, root=ds_root)
    policy, preprocessor, postprocessor = load_smolvla_pipeline(args.pretrained_path, device, ds_meta=ds_meta)

    # Resolve task prompt
    if args.task:
        task_prompt = args.task
    else:
        task_prompt = ds_meta.tasks.index[0] if hasattr(ds_meta, 'tasks') and len(ds_meta.tasks) > 0 else "perform the task"
    logger.info(f"Task prompt: '{task_prompt}'")

    # Parse and connect cameras
    cameras_config = parse_cameras_config(args.cameras)
    if not cameras_config:
        raise ValueError("No cameras configured")

    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Connected: {cam_name}")

    if not piper_robot:
        logger.error("Camera mode requires --piper (real robot) for observation state. "
                     "Use --dataset_root for offline evaluation instead.")
        return

    # Reset pipeline state
    policy.reset()
    for step in preprocessor.steps:
        if hasattr(step, 'reset'):
            step.reset()

    # --- Record starting EE pose (reference frame for ee_link-frame deltas) ---
    T_start = None
    if placo_viz:
        start_robot = placo_viz["robot"]
    elif fk_robot:
        start_robot = fk_robot
    else:
        start_robot = None
    if start_robot and piper_robot:
        for jn, val in zip(ARM_JOINTS, HOME_POSE_DEG):
            start_robot.set_joint(jn, np.deg2rad(val))
        start_robot.update_kinematics()
        T_start = start_robot.get_T_world_frame("ee_link").copy()
        logger.info(f"Reference EE pose (T_start): pos=({T_start[0,3]*1000:.1f}, "
                    f"{T_start[1,3]*1000:.1f}, {T_start[2,3]*1000:.1f}) mm")

    obs_state = None  # will be set from real robot FK each step
    image_history = {}
    _has_gui = True  # will be set to False if cv2.imshow fails (headless)
    step_count = 0
    ee_trail = []
    MAX_TRAIL = 500

    logger.info(f"\n{'='*60}")
    if args.execute:
        logger.info("  *** INTERACTIVE EXECUTION MODE ***")
        logger.info(f"  Commands: [e]xecute {args.n_execute_steps} steps / [r]e-infer / [q]uit")
        logger.info("  Ctrl+C = emergency stop (hold position)")
    else:
        logger.info("  Inference loop started. Press 'q' (with cameraview) or Ctrl+C to stop.")
    logger.info(f"{'='*60}\n")

    _quit = False
    try:
        while not _quit and (args.max_steps == 0 or step_count < args.max_steps):
            t0 = time.perf_counter()

            # Read real Piper state if connected
            if piper_robot:
                try:
                    obs = piper_robot.get_observation()
                    joint_deg = np.array([
                        obs["joint_1.pos"], obs["joint_2.pos"],
                        obs["joint_3.pos"], obs["joint_4.pos"],
                        obs["joint_5.pos"], obs["joint_6.pos"],
                    ])
                    gripper_val = obs.get("gripper.pos", 1.0)
                except Exception:
                    joint_deg = np.zeros(6)
                    gripper_val = 1.0

                # FK to get EE pose (use placo_viz robot, or standalone fk_robot)
                robot = placo_viz["robot"] if placo_viz else fk_robot
                if robot:
                    for jn, val in zip(ARM_JOINTS, joint_deg):
                        robot.set_joint(jn, np.deg2rad(val))
                    robot.update_kinematics()
                    T_ee = robot.get_T_world_frame("ee_link")

                    # Convert to ee_link-frame delta from T_start (matches training convention)
                    if T_start is not None:
                        obs_state = compute_ee_delta_state(T_ee, T_start, gripper_val)
                    else:
                        obs_state = pose_to_ee_state(T_ee, gripper_val)

                    if placo_viz:
                        # Update placo viz with real joint state
                        placo_viz["viz"].display(robot.state.q)

                        # Update link frames to follow real joints
                        for f in ALL_LINK_NAMES:
                            try:
                                T = robot.get_T_world_frame(f)
                                placo_viz["frame_viz"](f, T)
                            except Exception:
                                pass

                        # EE trail (green dots — real robot path history)
                        try:
                            ee_trail.append(T_ee[:3, 3].tolist())
                            if len(ee_trail) > MAX_TRAIL:
                                ee_trail = ee_trail[-MAX_TRAIL:]
                            placo_viz["points_viz"]("ee_trail", ee_trail, radius=0.003, color=0x00FF88)
                        except Exception:
                            pass
                else:
                    # No robot for FK — skip this inference step
                    logger.warning("No FK robot available, skipping inference step")
                    step_count += 1
                    continue

            # Build batch
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
                    horizon=1,  # SmolVLA uses single image per camera
                )
                image_history[cam_name] = hist_list

                batch[f"observation.images.{cam_name}"] = (
                    torch.from_numpy(temporal_img).unsqueeze(0).to(device)
                )

            # Inference
            t_infer = time.perf_counter()
            with torch.no_grad():
                processed = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed)
                pred_abs = postprocessor(pred_actions)  # (1, 50, 7) absolute

            pred_np = pred_abs[0].cpu().numpy()  # (50, 7) — ee_link-frame deltas
            infer_time = (time.perf_counter() - t_infer) * 1000

            # Convert predictions from ee_link-frame delta to world frame for viz/IK
            pred_world = np.zeros_like(pred_np)
            if T_start is not None:
                for t in range(len(pred_np)):
                    T_w = delta_action_to_world_pose(pred_np[t], T_start)
                    pos = T_w[:3, 3]
                    R = T_w[:3, :3]
                    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
                    if ang < 1e-10:
                        rv = np.zeros(3)
                    else:
                        ax = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
                        ax = ax / (2 * np.sin(ang))
                        rv = ax * ang
                    pred_world[t] = [*pos, *rv, pred_np[t, 6]]
            else:
                pred_world = pred_np.copy()

            # Display
            t0_action = pred_world[0]
            print(
                f"  step {step_count:4d} | "
                f"infer: {infer_time:5.0f}ms | "
                f"ee_xyz=[{t0_action[0]:.4f}, {t0_action[1]:.4f}, {t0_action[2]:.4f}] | "
                f"ee_rot=[{t0_action[3]:.4f}, {t0_action[4]:.4f}, {t0_action[5]:.4f}] | "
                f"grip={t0_action[6]:.3f} | "
                f"action_range=[{pred_world[:,:6].min():.4f}, {pred_world[:,:6].max():.4f}]"
            )

            # Update placo meshcat visualization (world frame)
            if placo_viz and step_count % max(1, args.placo_viz_interval) == 0:
                try:
                    update_placo_viz(placo_viz, pred_world, step_count)
                except Exception as e:
                    logger.warning(f"placo viz update error: {e}")

            # Show camera feed if requested (non-blocking)
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
                    if key == ord("q") or key == 27:  # 27 = ESC
                        logger.info("Exit key pressed")
                        _quit = True

            step_count += 1

            # --- Interactive execution mode ---
            if args.execute and piper_robot and kin is not None:
                import select
                prompt = (f"  [e]xecute {args.n_execute_steps} steps / "
                          f"[r]e-infer / [q]uit: ")
                cmd = None
                while cmd is None:
                    # Update placo with real robot state while waiting
                    robot_viz = placo_viz["robot"] if placo_viz else fk_robot
                    if robot_viz and piper_robot:
                        try:
                            obs_live = piper_robot.get_observation()
                            jlive = np.array([obs_live[f"joint_{i+1}.pos"] for i in range(6)])
                            for jn, val in zip(ARM_JOINTS, jlive):
                                robot_viz.set_joint(jn, np.deg2rad(val))
                            robot_viz.update_kinematics()
                            if placo_viz:
                                placo_viz["viz"].display(robot_viz.state.q)
                                for f in ALL_LINK_NAMES:
                                    try:
                                        T = robot_viz.get_T_world_frame(f)
                                        placo_viz["frame_viz"](f, T)
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    # Non-blocking input with 100ms timeout
                    sys.stdout.write(prompt)
                    sys.stdout.flush()
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        cmd = sys.stdin.readline().strip().lower()
                    else:
                        continue

                if cmd == "e":
                    try:
                        logger.info(f"  Executing {args.n_execute_steps} steps...")
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
                        logger.info("  Execution complete. Re-inferring...")
                    except KeyboardInterrupt:
                        logger.warning("  EMERGENCY STOP during execution")
                        emergency_hold(piper_robot, joint_deg, gripper_val)
                        raise
                    except Exception as e:
                        logger.error(f"  Execution error: {e}")
                        emergency_hold(piper_robot, joint_deg, gripper_val)
                elif cmd == "q":
                    _quit = True
                elif cmd == "r":
                    logger.info("  Re-inferring...")
                else:
                    logger.info("  Unknown command. Use e/r/q.")
            else:
                # Auto mode: maintain FPS
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, 1.0 / args.fps - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if piper_robot:
            try:
                obs = piper_robot.get_observation()
                hold_joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
                hold_gripper = obs.get("gripper.pos", 0.5)
                emergency_hold(piper_robot, hold_joints, hold_gripper)
                logger.info("Emergency hold applied")
            except Exception:
                pass
    finally:
        if args.cameraview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        for camera in cameras.values():
            camera.disconnect()
        if piper_robot:
            try:
                piper_robot.disconnect()
                logger.info("Piper disconnected")
            except Exception:
                pass
        logger.info(f"Done. Ran {step_count} steps.")


def run_dataset_mode(args):
    """Dry-run inference using dataset frames (no cameras needed)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.dataset_tools import recompute_stats

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    dataset_root = str(args.dataset_root)
    repo_id = Path(dataset_root).name

    # Load dataset to get metadata
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)

    # Recompute stats to match training
    ds = LeRobotDataset(repo_id, root=dataset_root)
    ds = recompute_stats(
        ds, num_workers=2,
        relative_action=True, relative_exclude_joints=["gripper"],
        relative_state=True, relative_exclude_state_joints=["gripper"],
        state_obs_steps=2, derive_state_from_action=True,
    )
    ds_meta = ds.meta

    policy, preprocessor, postprocessor = load_smolvla_pipeline(args.pretrained_path, device, ds_meta=ds_meta)

    # Resolve delta timestamps for multi-timestep batches
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    cfg = SmolVLAConfig(
        derive_state_from_action=True,
        use_relative_actions=True,
        device=device,
        resize_imgs_with_padding=(512, 512),
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        load_vlm_weights=False,
        push_to_hub=False,
    )
    dt = resolve_delta_timestamps(cfg, ds_meta)
    ds_dt = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=dt)

    logger.info(f"Dataset: {ds_meta.total_episodes} episodes, {ds_meta.total_frames} frames")
    logger.info(f"Action dims: {ds_meta.features['action']['names']}")

    # Reset pipeline
    policy.reset()
    for step in preprocessor.steps:
        if hasattr(step, 'reset'):
            step.reset()

    dim_names = ds_meta.features["action"]["names"]
    if isinstance(dim_names, dict):
        dim_names = dim_names.get("axes", list(dim_names.keys()))

    max_steps = min(args.max_steps if args.max_steps > 0 else ds_dt.num_frames, ds_dt.num_frames)
    logger.info(f"Running {max_steps} steps...\n")

    total_infer_time = 0.0
    for i in range(max_steps):
        raw = ds_dt[i]
        batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
                 for k, v in raw.items()}
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        gt_action = batch["action"].clone()

        t0 = time.perf_counter()
        with torch.no_grad():
            processed = preprocessor(batch)
            pred_actions = policy.predict_action_chunk(processed)
            pred_abs = postprocessor(pred_actions)
        infer_time = (time.perf_counter() - t0) * 1000
        total_infer_time += infer_time

        pred_np = pred_abs[0].cpu().numpy()  # (50, 7)
        gt_np = gt_action[0].cpu().numpy()  # (51, 7) — 51 frames before DeriveState

        if i % max(1, max_steps // 5) == 0 or i < 3:
            t0_pred = pred_np[0]
            t0_gt = gt_np[1]  # action[1] = first action after DeriveState trim
            err = np.abs(t0_pred - t0_gt)
            print(f"  frame {i:4d} | infer: {infer_time:5.0f}ms | "
                  f"pred_ee=[{t0_pred[0]:.4f},{t0_pred[1]:.4f},{t0_pred[2]:.4f}] | "
                  f"gt_ee=[{t0_gt[0]:.4f},{t0_gt[1]:.4f},{t0_gt[2]:.4f}] | "
                  f"err_xyz=[{err[0]:.4f},{err[1]:.4f},{err[2]:.4f}]")

    logger.info(f"\nDone. {max_steps} steps, avg inference: {total_infer_time/max_steps:.0f}ms")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SmolVLA UMI EE-pose live inference")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to trained checkpoint directory")
    parser.add_argument("--fps", type=int, default=FPS,
                        help="Control loop frequency")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max steps (0=infinite for camera, all for dataset)")
    parser.add_argument("--cameras", type=str, default=None,
                        help="Camera YAML config (enables camera mode)")
    parser.add_argument("--cameraview", action="store_true",
                        help="Show camera feed window")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Dataset root path (enables dataset dry-run mode)")
    parser.add_argument("--placo_viz", action="store_true",
                        help="Enable placo meshcat visualization of predicted EE actions on Piper URDF")
    parser.add_argument("--placo_viz_interval", type=int, default=3,
                        help="Update meshcat every N steps (default: 3, to avoid overloading)")
    parser.add_argument("--piper", type=str, default=None,
                        help="CAN interface for real Piper (e.g. can0), enables real robot mode")
    parser.add_argument("--execute", action="store_true",
                        help="Enable interactive execution mode (requires --piper)")
    parser.add_argument("--n_execute_steps", type=int, default=8,
                        help="Number of action steps to execute per 'e' command (default: 8)")
    parser.add_argument("--position_weight", type=float, default=1.0,
                        help="IK position weight (default: 1.0)")
    parser.add_argument("--orientation_weight", type=float, default=0.01,
                        help="IK orientation weight (default: 0.01)")
    parser.add_argument("--task", type=str, default=None,
                        help="Task prompt for the policy (default: read from dataset)")
    args = parser.parse_args()

    if args.execute and not args.piper:
        parser.error("--execute requires --piper (CAN interface for real robot)")

    if args.cameras:
        run_camera_mode(args)
    elif args.dataset_root:
        run_dataset_mode(args)
    else:
        parser.error("Either --cameras or --dataset_root required")


if __name__ == "__main__":
    main()
