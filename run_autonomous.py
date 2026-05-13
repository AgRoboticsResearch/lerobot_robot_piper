#!/usr/bin/env python
"""Autonomous SmolVLA UMI-EE control with timestamp-synchronized pipeline.

Architecture (2 processes + inference thread + thin main loop):

  Camera (30fps) → ObservationSynchronizer → InferenceThread (3-5Hz)
  InferenceThread → schedule_waypoint commands → SmoothController (50Hz IK, mp.Process) → Piper
  Controller → SharedMemoryRingBuffer (state) → InferenceThread reads ActualEEPose

Main thread: keyboard handler + stats logging (20Hz UI).

Usage:
  conda activate lerobot_piper_sroi && python lerobot_robot_piper/run_autonomous.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --cameras "{ color: {type: intelrealsense, serial_number_or_name: '230322274337', width: 640, height: 480, fps: 30} }" \
      --piper can0
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyboard reader (from v2)
# ---------------------------------------------------------------------------

class _KeyboardReader:
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
            self._started = True

    def restore(self):
        if self._old_term is not None and self._fd is not None:
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            except Exception:
                pass
            self._old_term = None

    def read(self):
        if not self._started:
            self.start()
        if self._fd is None:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                return (sys.stdin.readline().strip().lower() or None)
            return None
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
    return _kb.read()


# ---------------------------------------------------------------------------
# Camera config parsing (from v2)
# ---------------------------------------------------------------------------

def parse_cameras_config(cameras_str: str) -> dict:
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


# ---------------------------------------------------------------------------
# SmolVLA pipeline loading (from v2)
# ---------------------------------------------------------------------------

def load_smolvla_pipeline(pretrained_path: str, device: str = "cuda", dataset_root: str | None = None):
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
    if dataset_root:
        ds_root = dataset_root
    else:
        ds_root = ds_cfg.get("root", str(PROJECT_ROOT / "Datasets" / ds_repo_id))

    logger.info(f"Loading stats from training dataset: {ds_repo_id} at {ds_root}")
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

    task_prompt = None
    if hasattr(ds_meta, "tasks") and len(ds_meta.tasks) > 0:
        task_prompt = ds_meta.tasks.index[0]
    if not task_prompt:
        task_prompt = "perform the task"
    logger.info(f"Task prompt: '{task_prompt}'")

    return policy, preprocessor, postprocessor, task_prompt


# ---------------------------------------------------------------------------
# Hardware connection
# ---------------------------------------------------------------------------

def connect_piper(args):
    from lerobot_robot_piper.piper import Piper
    from lerobot_robot_piper.config_piper import PiperConfig

    logger.info(f"Connecting to Piper on {args.piper}...")
    config = PiperConfig(
        can_interface=args.piper,
        include_gripper=True,
        use_degrees=True,
        cameras={},
    )
    piper = Piper(config)
    piper.connect()
    logger.info("Piper connected and enabled")
    return piper


def connect_cameras(args):
    from lerobot.cameras import make_cameras_from_configs

    cameras_config = parse_cameras_config(args.cameras)
    if not cameras_config:
        raise ValueError("No cameras configured")

    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Camera connected: {cam_name}")
    return cameras


def move_to_home(piper, home_pose_deg):
    logger.info("Moving to home pose...")
    obs = piper.get_observation()
    current = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
    n_steps = 50
    for step in range(n_steps + 1):
        alpha = step / n_steps
        target = current * (1 - alpha) + home_pose_deg * alpha
        action = {f"joint_{i+1}.pos": target[i] for i in range(6)}
        action["gripper.pos"] = 1.0
        piper.send_action(action)
        time.sleep(0.05)
    logger.info("Home pose reached")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous SmolVLA UMI-EE control")

    # Hardware
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--cameras", type=str, required=True)
    parser.add_argument("--piper", type=str, default="can0")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    # Control tuning
    parser.add_argument("--control_hz", type=float, default=50.0)
    parser.add_argument("--max_vel_deg_s", type=float, default=60.0)
    parser.add_argument("--position_weight", type=float, default=1.0)
    parser.add_argument("--orientation_weight", type=float, default=0.01)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--execution_latency", type=float, default=0.0,
                        help="Execution latency compensation (seconds, from bench_piper)")
    parser.add_argument("--max_pos_speed", type=float, default=0.01,
                        help="Max EE position speed (m/s). Default: 0.01 = 1cm/s")
    parser.add_argument("--max_rot_speed", type=float, default=0.5,
                        help="Max EE rotation speed (rad/s). Default: 0.5")

    # Pipeline tuning
    parser.add_argument("--buffer_threshold", type=int, default=15,
                        help="Trigger inference when buffer remaining < this")
    parser.add_argument("--similarity_atol", type=float, default=1.0,
                        help="Observation similarity threshold (0=disabled)")
    parser.add_argument("--n_steps", type=int, default=0,
                        help="Max inference steps (0=infinite)")

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    device = args.device if args.device else ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    # 1. Load policy
    logger.info("Loading SmolVLA pipeline...")
    policy, preprocessor, postprocessor, task_prompt = load_smolvla_pipeline(
        args.pretrained_path, device, dataset_root=args.dataset_root,
    )
    if args.task:
        task_prompt = args.task

    # 2. Connect hardware (for joint reader + home move only)
    piper = connect_piper(args)
    cameras = connect_cameras(args)

    # 3. Create shared components
    from multiprocessing.managers import SharedMemoryManager
    from controller.observation_synchronizer import ObservationSynchronizer
    from controller.inference_thread import InferenceThread
    from controller.smooth_controller import SmoothController

    # 4. Move to home (using main process Piper)
    move_to_home(piper, HOME_POSE_DEG)

    # 5. Create controller process (mp.Process with shared memory IPC)
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
        frequency=args.control_hz,
        max_vel_deg_s=args.max_vel_deg_s,
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
    )

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
        similarity_atol=args.similarity_atol,
    )

    # 6. Wait for user to press 's' to start
    logger.warning("=" * 50)
    logger.warning("  Robot is at HOME pose. Motors ENABLED.")
    logger.warning("  Press 's' to START autonomous control")
    logger.warning("  Press 'q' to QUIT without starting")
    logger.warning("=" * 50)

    _started = False
    while not _started:
        key = _check_key()
        if key == "s":
            _started = True
        elif key == "q":
            logger.info("Quit before start")
            for cam in cameras.values():
                try:
                    cam.disconnect()
                except Exception:
                    pass
            try:
                piper.disconnect()
            except Exception:
                pass
            return
        time.sleep(0.05)

    # 7. Start controller process + inference thread
    logger.info("Starting controller process and inference...")
    controller.start(wait=True)
    inference.start()

    # 8. Main loop: thin UI
    logger.info("Autonomous control running. Keys: f=freeze/unfreeze, q=quit")
    _frozen = False
    last_log_time = 0.0

    try:
        while True:
            key = _check_key()

            if key == "f":
                _frozen = not _frozen
                if _frozen:
                    logger.warning("*** FROZEN — press 'f' to unfreeze ***")
                    controller.input_queue.clear()
                else:
                    logger.info("*** UNFROZEN ***")

            elif key == "q":
                logger.info("Quit requested")
                break

            # Periodic stats logging
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                last_log_time = now
                inf_stats = inference.get_stats()
                queue_remaining = controller.remaining()
                try:
                    state = controller.get_state()
                    ctrl_q = int(state["ActualJointState"][0]) if len(state["ActualJointState"]) > 0 else 0
                except Exception:
                    ctrl_q = 0
                logger.info(
                    f"inf_steps={inf_stats['steps']} "
                    f"errors={inf_stats['errors']} "
                    f"skipped={inf_stats['skipped']} "
                    f"infer={inf_stats['last_infer_ms']:.0f}ms | "
                    f"queue={queue_remaining}"
                )

            time.sleep(0.05)  # 20Hz UI loop

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
        try:
            piper.disconnect()
        except Exception:
            pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
