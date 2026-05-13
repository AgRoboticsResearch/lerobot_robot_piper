#!/usr/bin/env python
"""Visualize SmolVLA waypoints with placo meshcat (dry_run mode).

Runs the full pipeline (inference + controller + shared memory) but does NOT
send motor commands. The robot can be moved by hand. Visualizes:
  - URDF at live sensor positions (green)
  - Pending waypoints from Queue as colored points (green→red)
  - Current EE position (blue frame)
  - Stats in terminal

Usage:
  conda activate lerobot_piper_sroi && python lerobot_robot_piper/viz_waypoints_placo.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322273077', width: 640, height: 480, fps: 30} }" \
      --piper can0
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

URDF_SRC = str(
    PROJECT_ROOT / "lerobot_robot_piper" / "lerobot_robot_piper" / "urdf" / "piper_description.urdf"
)
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_POSE_DEG = np.array([0.0, 61.0, -13.0, 0.0, -36.0, 0.0])
ALL_LINK_NAMES = [
    "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
    "ee_link", "camera_link",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Placo visualization
# ---------------------------------------------------------------------------

def init_placo_viz():
    import placo
    from placo_utils.visualization import robot_viz, frame_viz, points_viz

    robot = placo.RobotWrapper(URDF_SRC)
    for jn, val in zip(ARM_JOINTS, HOME_POSE_DEG):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    T_world_ee = robot.get_T_world_frame("ee_link")

    viz = robot_viz(robot)
    viz.display(robot.state.q)
    time.sleep(1.0)

    for frame_name in ALL_LINK_NAMES:
        try:
            T = robot.get_T_world_frame(frame_name)
            frame_viz(frame_name, T)
        except Exception:
            pass

    logger.info(f"meshcat viewer: http://127.0.0.1:7001/static/")
    logger.info(
        f"Initial EE: pos=({T_world_ee[0,3]*1000:.0f}, "
        f"{T_world_ee[1,3]*1000:.0f}, {T_world_ee[2,3]*1000:.0f}) mm"
    )

    return {
        "robot": robot,
        "viz": viz,
        "frame_viz": frame_viz,
        "points_viz": points_viz,
    }


def update_robot_viz(viz_state, joint_deg):
    from placo_utils.visualization import frame_viz

    robot = viz_state["robot"]
    viz = viz_state["viz"]
    fv = viz_state["frame_viz"]

    for jn, val in zip(ARM_JOINTS, joint_deg):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()
    viz.display(robot.state.q)

    for f in ALL_LINK_NAMES:
        try:
            T = robot.get_T_world_frame(f)
            fv(f, T)
        except Exception:
            pass


def viz_waypoints(viz_state, controller):
    """Visualize pending waypoints from SharedMemoryQueue (peek, no consume)."""
    from placo_utils.visualization import frame_viz as fv, points_viz as pv
    from queue import Empty

    try:
        commands = controller.input_queue.peek_all()
    except Empty:
        # Clear old segments
        for s in range(5):
            pv(f"wp_seg{s}", np.zeros((1, 3)), radius=0.001, color=0x444444)
        fv("wp_start", np.eye(4))
        fv("wp_end", np.eye(4))
        return

    n = len(commands["cmd"])
    poses = commands["target_pose"]  # (n, 6)
    points = poses[:, :3]  # (n, 3)

    # Color: green (near) → yellow → red (far)
    n_segments = min(5, n)
    seg_len = max(1, n // n_segments)
    colors = [0x00FF00, 0x88FF00, 0xFFFF00, 0xFF8800, 0xFF0000]
    for s in range(n_segments):
        start = s * seg_len
        end = min((s + 1) * seg_len + 1, n)
        seg = points[start:end]
        if len(seg) > 0:
            pv(f"wp_seg{s}", seg, radius=0.004, color=colors[min(s, len(colors) - 1)])
    for s in range(n_segments, 5):
        pv(f"wp_seg{s}", np.zeros((1, 3)), radius=0.001, color=0x444444)

    # Start/end frames
    fv("wp_start", _pose6d_to_matrix(poses[0]))
    if n > 1:
        fv("wp_end", _pose6d_to_matrix(poses[-1]))


def _pose6d_to_matrix(pose_6d):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, 3] = pose_6d[:3]
    T[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    return T


# ---------------------------------------------------------------------------
# Config / pipeline helpers
# ---------------------------------------------------------------------------

def parse_cameras_config(cameras_str):
    from lerobot.cameras import CameraConfig
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


def load_smolvla_pipeline(pretrained_path, device="cuda", dataset_root=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    pretrained_path = str(pretrained_path)
    config_path = Path(pretrained_path) / "train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        policy_config = train_config.get("policy", {})
        ds_cfg = train_config.get("dataset", {})
    else:
        policy_config = {}
        ds_cfg = {}

    ds_repo_id = ds_cfg.get("repo_id", "sroi_piper_strawberry_picking")
    ds_root = dataset_root or ds_cfg.get("root", str(PROJECT_ROOT / "Datasets" / ds_repo_id))

    logger.info(f"Loading stats from: {ds_repo_id} at {ds_root}")
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    ds_meta = LeRobotDatasetMetadata(ds_repo_id, root=ds_root)

    cfg = SmolVLAConfig(
        derive_state_from_action=policy_config.get("derive_state_from_action", True),
        use_relative_actions=policy_config.get("use_relative_actions", True),
        relative_exclude_joints=policy_config.get("relative_exclude_joints", ["gripper"]),
        relative_exclude_state_joints=policy_config.get("relative_exclude_state_joints", ["gripper"]),
        device=device,
        resize_imgs_with_padding=tuple(policy_config.get("resize_imgs_with_padding", (512, 512))),
        freeze_vision_encoder=policy_config.get("freeze_vision_encoder", True),
        train_expert_only=policy_config.get("train_expert_only", True),
        train_state_proj=policy_config.get("train_state_proj", True),
        load_vlm_weights=False, push_to_hub=False,
        pretrained_path=pretrained_path,
    )

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=pretrained_path, dataset_stats=ds_meta.stats,
    )

    task_prompt = None
    if hasattr(ds_meta, "tasks") and len(ds_meta.tasks) > 0:
        task_prompt = ds_meta.tasks.index[0]
    if not task_prompt:
        task_prompt = "perform the task"

    return policy, preprocessor, postprocessor, task_prompt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Visualize SmolVLA waypoints (dry_run)")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--cameras", type=str, required=True)
    parser.add_argument("--piper", type=str, default="can0")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--buffer_threshold", type=int, default=15)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    device = args.device
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # 1. Load policy
    logger.info("Loading SmolVLA pipeline...")
    policy, preprocessor, postprocessor, task_prompt = load_smolvla_pipeline(
        args.pretrained_path, device, dataset_root=args.dataset_root,
    )
    if args.task:
        task_prompt = args.task
    logger.info(f"Task prompt: '{task_prompt}'")

    # 2. Init placo visualization
    placo_viz = init_placo_viz()

    # 3. Connect cameras
    from lerobot.cameras import make_cameras_from_configs
    cameras_config = parse_cameras_config(args.cameras)
    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Camera connected: {cam_name}")

    # 4. Create controller process (dry_run — no motor commands)
    from multiprocessing.managers import SharedMemoryManager
    from controller.smooth_controller import SmoothController
    from controller.observation_synchronizer import ObservationSynchronizer
    from controller.inference_thread import InferenceThread

    piper_config = {
        "can_interface": args.piper,
        "include_gripper": True,
        "use_degrees": True,
        "cameras": {},
    }

    shm_manager = SharedMemoryManager()
    shm_manager.start()

    controller = SmoothController(
        shm_manager=shm_manager,
        piper_config=piper_config,
        urdf_path=URDF_SRC,
        target_frame="ee_link",
        joint_names=ARM_JOINTS,
        frequency=50.0,
        dry_run=True,  # No send_action, but full pipeline otherwise
        verbose=True,
    )
    controller.start(wait=True)
    logger.info("Controller process started (dry_run=True)")

    # 5. Create inference pipeline
    synchronizer = ObservationSynchronizer(controller, cameras)
    inference = InferenceThread(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        controller=controller,
        synchronizer=synchronizer,
        task_prompt=task_prompt,
        device=device,
        threshold=args.buffer_threshold,
        fps=args.fps,
        similarity_atol=0.0,  # Always infer in viz mode
    )
    inference.start()

    # 6. Main viz loop
    logger.info("Visualization running. Move arm by hand to see predicted waypoints.")
    logger.info("Press Ctrl+C to quit")
    logger.info("Open http://127.0.0.1:7001/static/ in browser")

    last_log_time = 0.0

    try:
        while True:
            # Update URDF from RingBuffer (real sensor state)
            try:
                state = controller.get_state()
                joints_now = state["ActualJointState"]
                update_robot_viz(placo_viz, joints_now)
            except Exception:
                pass

            # Visualize pending waypoints from Queue (peek, no consume)
            viz_waypoints(placo_viz, controller)

            # Periodic stats
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                last_log_time = now
                inf_stats = inference.get_stats()
                queue_n = controller.remaining()

                # EE position from placo
                try:
                    T_ee = placo_viz["robot"].get_T_world_frame("ee_link")
                    ee_pos = T_ee[:3, 3] * 1000
                    ee_str = f"xyz=[{ee_pos[0]:.1f},{ee_pos[1]:.1f},{ee_pos[2]:.1f}]mm"
                except Exception:
                    ee_str = "xyz=?"

                logger.info(
                    f"inf_steps={inf_stats['steps']} "
                    f"errors={inf_stats['errors']} "
                    f"skipped={inf_stats['skipped']} "
                    f"infer={inf_stats['last_infer_ms']:.0f}ms | "
                    f"queue={queue_n} | "
                    f"{ee_str}"
                )

            time.sleep(0.05)  # 20Hz viz update

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        logger.info("Shutting down...")
        inference.stop()
        controller.stop()
        shm_manager.shutdown()
        for cam in cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        logger.info("Done")


if __name__ == "__main__":
    main()
