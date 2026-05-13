#!/usr/bin/env python
"""Real-hardware autonomous SmolVLA control (UMI-EE architecture).

Architecture (same as viz_waypoints_placo.py, but dry_run=False):

  Camera (30fps) → ObservationSynchronizer → InferenceThread (3-5Hz)
  InferenceThread → schedule_waypoint → SmoothController (50Hz IK, mp.Process) → Piper
  Controller → SharedMemoryRingBuffer (state) → InferenceThread reads ActualEEPose

  Main thread: move-to-home → keyboard handler + stats (20Hz UI)

CAN bus lifecycle (single connection — controller subprocess only):
  1. Controller subprocess connects Piper → EnablePiper → motors active
  2. Main process sends move_to_joints(home) through SharedMemoryQueue
  3. No dual-connection, no CAN bus gap, no gravity drop

Usage:
  conda activate lerobot_piper_sroi && python lerobot_robot_piper/run_autonomous_v2.py \
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
HOME_POSE_DEG = np.array([0.0, 50.60, -50.40, -1.21, 10.00, 0.00])

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyboard reader
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
                return sys.stdin.readline().strip().lower() or None
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


# ---------------------------------------------------------------------------
# Config helpers (shared with viz_waypoints_placo.py)
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

    logger.info(f"Loading stats from: {ds_repo_id} at {ds_root}")
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
# Home pose — FK computed in main, move_to_pose sent through controller
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-hardware autonomous SmolVLA control (UMI-EE architecture)"
    )

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
    parser.add_argument("--max_pos_speed", type=float, default=0.01,
                        help="Max EE position speed (m/s)")
    parser.add_argument("--max_rot_speed", type=float, default=0.5,
                        help="Max EE rotation speed (rad/s)")

    # Pipeline tuning
    parser.add_argument("--buffer_threshold", type=int, default=15)
    parser.add_argument("--similarity_atol", type=float, default=1.0)
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

    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ── 1. Load policy ──────────────────────────────────────────────
    logger.info("Loading SmolVLA pipeline...")
    policy, preprocessor, postprocessor, task_prompt = load_smolvla_pipeline(
        args.pretrained_path, device, dataset_root=args.dataset_root,
    )
    if args.task:
        task_prompt = args.task
    logger.info(f"Task prompt: '{task_prompt}'")

    # ── 2. Connect cameras ──────────────────────────────────────────
    from lerobot.cameras import make_cameras_from_configs
    cameras_config = parse_cameras_config(args.cameras)
    cameras = make_cameras_from_configs(cameras_config)
    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Camera connected: {cam_name}")

    # ── 3. Start controller subprocess ──────────────────────────────
    #    Controller creates its own Piper connection (EnablePiper → motors active).
    #    No main-process Piper needed at all — home move goes through
    #    SharedMemoryQueue via move_to_joints (direct joint-space, no IK).
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
        frequency=args.control_hz,
        max_vel_deg_s=args.max_vel_deg_s,
        position_weight=args.position_weight,
        orientation_weight=args.orientation_weight,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
    )

    logger.info("Starting controller subprocess...")
    controller.start(wait=True)
    logger.info("Controller subprocess ready (Piper enabled, motors active)")

    # ── 4. Move to home via direct joint command (no IK) ────────────
    logger.info(f"Moving to home pose via move_to_joints (duration=3s)...")
    controller.move_to_joints(HOME_POSE_DEG, duration=3.0, gripper=1.0)
    time.sleep(3.5)
    logger.info("Home pose reached")

    # ── 5. Create inference pipeline ────────────────────────────────
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

    # ── 6. Wait for user confirmation ───────────────────────────────
    logger.warning("=" * 55)
    logger.warning("  Robot is at HOME pose. Motors ENABLED.")
    logger.warning("  Press 's' to START autonomous inference")
    logger.warning("  Press 'q' to QUIT")
    logger.warning("=" * 55)

    _started = False
    while not _started:
        key = _kb.read()
        if key == "s":
            _started = True
        elif key == "q":
            logger.info("Quit before start")
            for cam in cameras.values():
                try:
                    cam.disconnect()
                except Exception:
                    pass
            shm_manager.shutdown()
            return
        time.sleep(0.05)

    # ── 7. Start inference ──────────────────────────────────────────
    inference.start()
    logger.info("InferenceThread started")

    # ── 8. Main loop: keyboard + stats ──────────────────────────────
    logger.info("Autonomous control ACTIVE. Keys: f=freeze/unfreeze, q=quit")
    _frozen = False
    last_log_time = 0.0

    try:
        while True:
            key = _kb.read()

            if key == "f":
                _frozen = not _frozen
                if _frozen:
                    logger.warning("*** FROZEN — clearing queue, press 'f' to unfreeze ***")
                    inference.stop()
                    controller.input_queue.clear()
                else:
                    logger.info("*** UNFROZEN — resuming inference ***")
                    inference.start()

            elif key == "q":
                logger.info("Quit requested")
                break

            # Periodic stats (1Hz)
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                last_log_time = now
                inf_stats = inference.get_stats()
                queue_n = controller.remaining()

                # EE position from RingBuffer
                ee_str = ""
                try:
                    state = controller.get_state()
                    joints = state["ActualJointState"]
                    ee_pose = state["ActualEEPose"]
                    ee_str = (f"xyz=[{ee_pose[0]*1000:.0f},"
                              f"{ee_pose[1]*1000:.0f},"
                              f"{ee_pose[2]*1000:.0f}]mm")
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

                if args.n_steps > 0 and inf_stats["steps"] >= args.n_steps:
                    logger.info(f"Reached n_steps limit ({args.n_steps})")
                    break

            time.sleep(0.05)  # 20Hz UI

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
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
