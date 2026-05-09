#!/usr/bin/env python

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Visualize predicted or GT trajectories projected onto camera images.

Supports two modes:

1. **Camera mode** (with --cameras): Connects to physical cameras, runs policy
   inference in a real-time loop, and optionally overlays predicted trajectories.

2. **Dataset mode** (with --dataset_root): Loads a RelativeEE dataset and projects
   trajectories onto the observation images, saving the result as MP4 videos.
   - Without --inference: projects GT action trajectories -> proj_gt_episode_N.mp4
   - With --inference: runs policy inference per frame -> proj_inference_episode_N.mp4

Usage (camera mode):
    python visualize_camera_prediction.py \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model \
        --cameras "{ front: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30, fourcc: MJPG} }"

Usage (dataset mode - GT):
    python visualize_camera_prediction.py \
        --dataset_root /mnt/data0/data/sroi/sroi_lab_picking \
        --episode_indices 0

Usage (dataset mode - inference):
    python visualize_camera_prediction.py \
        --dataset_root /mnt/data0/data/sroi/sroi_lab_picking \
        --episode_indices 0 \
        --inference \
        --pretrained_path outputs/train/my_model/checkpoints/001000/pretrained_model
"""

import json
import logging
import time
from pathlib import Path
from typing import Any
from threading import Thread
import os
import sys
import warnings

import numpy as np
import cv2
import torch
import yaml
import imageio.v3 as iio
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for real-time updates
import matplotlib.pyplot as plt

# Try to import pyrealsense2 for getting camera intrinsics
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.datasets.relative_ee_dataset import RelativeEEDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower.relative_ee_processor import pose10d_to_mat
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.temporal_wrapper import TemporalACTWrapper
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import init_logging

FPS = 30  # Control loop frequency (Hz) - should match training fps

logger = logging.getLogger(__name__)


def load_camera_matrix_from_dataset(dataset_root: str) -> np.ndarray:
    """Load camera intrinsics matrix K from dataset meta/camera_info.json.

    The file uses ROS CameraInfo format where K is a flat 9-element array:
    [fx, 0, cx, 0, fy, cy, 0, 0, 1]

    Args:
        dataset_root: Root path of the dataset

    Returns:
        (3, 3) camera intrinsics matrix K
    """
    camera_info_path = Path(dataset_root) / "meta" / "camera_info.json"
    if not camera_info_path.exists():
        raise FileNotFoundError(
            f"camera_info.json not found at {camera_info_path}. "
            f"Expected ROS CameraInfo format with 'K' field."
        )
    with open(camera_info_path) as f:
        camera_info = json.load(f)

    K_flat = camera_info["K"]
    camera_matrix = np.array(K_flat, dtype=np.float64).reshape(3, 3)
    logger.info(f"Loaded camera matrix from {camera_info_path}:")
    logger.info(f"  fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}, "
                f"cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}")
    return camera_matrix


def get_kinematic_transforms(urdf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Calculate transformations from optical to base camera loop and camera to physical gripper."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            kin_opt = RobotKinematics(str(urdf_path), target_frame_name="camera_optical_link")
            kin_cam = RobotKinematics(str(urdf_path), target_frame_name="camera_link")
            kin_grip = RobotKinematics(str(urdf_path), target_frame_name="gripper_frame_link")

            joints = np.zeros(len(kin_opt.joint_names))
            T_base_opt = kin_opt.forward_kinematics(joints)
            T_base_cam = kin_cam.forward_kinematics(joints)
            T_base_grip = kin_grip.forward_kinematics(joints)

            # View shift from camera_optical_link to camera_link
            T_opt_cam = np.linalg.inv(T_base_opt) @ T_base_cam
            
            # View shift from camera_link to gripper_frame_link
            T_cam_grip = np.linalg.inv(T_base_cam) @ T_base_grip
            
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            
    return T_opt_cam, T_cam_grip


def actions_to_3d_points(actions: np.ndarray, T_opt_cam: np.ndarray, T_cam_grip: np.ndarray) -> np.ndarray:
    """Convert relative 10D action chunk to 3D trajectory points in optical frame.

    Each action is a 10D relative pose [dx,dy,dz,rot6d_0..5,gripper] targeting camera_link.

    Args:
        actions: (N, 10) array of relative actions
        T_opt_cam: Transform from camera_optical_link to camera_link
        T_cam_grip: Transform from camera_link to gripper_frame_link

    Returns:
        (N+1, 3) array of 3D positions in optical frame (origin + one per action)
    """
    # Base pose in optical frame corresponds to the initial position of the gripper
    # Without any relative movement, the gripper in optical frame is just T_opt_cam @ T_cam_grip
    
    # We want to trace the gripper point.
    # The action specifies a relative transform applied to camera_link: T_rel
    # So the new camera_link pose in optical frame = T_opt_cam @ T_rel
    # The new gripper tip pose in optical frame = T_opt_cam @ T_rel @ T_cam_grip
    
    positions = []
    
    # Current/start gripper position in optical frame
    start_pose = T_opt_cam @ T_cam_grip
    positions.append(start_pose[:3, 3].copy())

    for action in actions:
        rel_pose = pose10d_to_mat(action[:9])
        future_gripper_optical = T_opt_cam @ rel_pose @ T_cam_grip
        positions.append(future_gripper_optical[:3, 3].copy())

    return np.array(positions)


def update_3d_trajectory_plot(ax, trajectory_3d, frame_idx, ep_idx, mode_label, trajectory_3d_gt=None, gripper=None, gripper_gt=None):
    """Update a matplotlib 3D axes with trajectory data."""
    ax.clear()
    
    if trajectory_3d_gt is not None and len(trajectory_3d_gt) > 1:
        ax.plot(trajectory_3d_gt[:, 0], trajectory_3d_gt[:, 1], trajectory_3d_gt[:, 2],
                'r--', linewidth=2, label='GT')
        ax.plot([trajectory_3d_gt[-1, 0]], [trajectory_3d_gt[-1, 1]], [trajectory_3d_gt[-1, 2]],
                'mo', markersize=6)
        if gripper_gt is not None:
            pts = trajectory_3d_gt[1:][gripper_gt < 0.1]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='m', marker='x', s=50, label='GT Grip')
                
    if len(trajectory_3d) > 1:
        ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2],
                'b-', linewidth=2, label='Pred' if trajectory_3d_gt is not None else 'Traj')
        ax.plot([trajectory_3d[0, 0]], [trajectory_3d[0, 1]], [trajectory_3d[0, 2]],
                'go', markersize=8, label='Start')
        ax.plot([trajectory_3d[-1, 0]], [trajectory_3d[-1, 1]], [trajectory_3d[-1, 2]],
                'ro', markersize=8, label='End (Pred)')
        if gripper is not None:
            pts = trajectory_3d[1:][gripper < 0.1]
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='k', marker='x', s=50, label='Pred Grip')
                
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Ep {ep_idx} Frame {frame_idx} - {mode_label.upper()} 3D Trajectory')
    ax.legend(loc='upper right')
    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_zlim(-0.25, 0.25)
    ax.view_init(elev=20, azim=45)


def render_figure_to_image(fig):
    """Render a matplotlib figure to an RGB numpy array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def project_points_to_image(points_3d: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D image coordinates using camera intrinsics.

    Args:
        points_3d: (N, 3) array of 3D points in camera optical frame (x right, y down, z forward)
        camera_matrix: (3, 3) camera intrinsics matrix K

    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    # Points are natively in optical frame via RobotKinematics (x right, y down, z forward)

    # Project using camera matrix: [u, v, 1]^T = K * [X, Y, Z]^T / Z
    points_homogeneous = points_3d  # (N, 3)
    z = points_homogeneous[:, 2:3]  # (N, 1)

    # Avoid division by zero
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    points_2d_homogeneous = (camera_matrix @ points_homogeneous.T).T  # (N, 3)
    points_2d = points_2d_homogeneous[:, :2] / z  # (N, 2)

    return points_2d


def draw_trajectory_on_image(img: np.ndarray, points_2d: np.ndarray, cmap: str = "pred", gripper: np.ndarray = None) -> np.ndarray:
    """Draw trajectory on image.

    Args:
        img: (H, W, 3) RGB image
        points_2d: (N, 2) array of 2D pixel coordinates
        cmap: Color mapping ('pred' for default blue/green, 'gt' for orange/yellow)
        gripper: (N,) array of gripper states (0 = closed, 1 = open)

    Returns:
        Image with trajectory drawn
    """
    img_draw = img.copy()

    n = len(points_2d)
    if n == 0:
        return img_draw

    if cmap == "gt":
        # Green to Red for GT
        colors = [(0, int(255 * (1 - i / max(1, n - 1))), 255) for i in range(max(1, n - 1))]
    else:
        # Gradient from blue to green
        colors = [
            (int(255 * i / max(1, n - 1)), int(255 * (1 - i / max(1, n - 1))), 0)  # BGR
            for i in range(max(1, n - 1))
        ]

    # Draw line segments
    for i in range(len(points_2d) - 1):
        pt1 = tuple(points_2d[i].astype(int))
        pt2 = tuple(points_2d[i + 1].astype(int))

        # Check if points are within image bounds
        h, w = img.shape[:2]
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(img_draw, pt1, pt2, colors[i % len(colors)], 2)
            if gripper is not None and gripper[i] < 0.1:
                color = (0, 0, 255) if cmap == "gt" else (255, 0, 0)
                cv2.drawMarker(img_draw, pt1, color, markerType=cv2.MARKER_CROSS, markerSize=1, thickness=2)

    # Draw start point (green) and end point (red)
    if len(points_2d) > 0:
        start_pt = tuple(points_2d[0].astype(int))
        end_pt = tuple(points_2d[-1].astype(int))
        h, w = img.shape[:2]

        if 0 <= start_pt[0] < w and 0 <= start_pt[1] < h:
            cv2.circle(img_draw, start_pt, 5, (0, 255, 0), -1)  # Green
        if 0 <= end_pt[0] < w and 0 <= end_pt[1] < h:
            cv2.circle(img_draw, end_pt, 5, (0, 0, 255), -1)  # Red

    return img_draw


class TrajectoryPlotter:
    """Real-time 3D trajectory plotter using matplotlib."""

    def __init__(self):
        """Initialize the trajectory plotter."""
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_z = []

        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initialize plot line and markers
        self.line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Predicted Trajectory')
        self.start, = self.ax.plot([], [], [], 'go', markersize=5, label='Start')
        self.head, = self.ax.plot([], [], [], 'ro', markersize=5, label='End Position')

        # Set labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Predicted End-Effector Trajectory (from origin)')
        self.ax.legend()

        # Set fixed axis limits: all from -0.25 to +0.25
        self.ax.set_xlim(-0.25, 0.25)
        self.ax.set_ylim(-0.25, 0.25)
        self.ax.set_zlim(-0.25, 0.25)

        # Start matplotlib in non-blocking mode
        plt.ion()
        self.fig.show()
        self.fig.canvas.flush_events()

    def set_trajectory(self, trajectory_3d: np.ndarray):
        """Set the trajectory from pre-calculated 3D positions in optical frame.

        Args:
            trajectory_3d: (N, 3) matrix of 3D points
        """
        self.trajectory_x = trajectory_3d[:, 0]
        self.trajectory_y = trajectory_3d[:, 1]
        self.trajectory_z = trajectory_3d[:, 2]

    def update(self):
        """Update the plot with current trajectory."""
        # Update line data
        self.line.set_data(self.trajectory_x, self.trajectory_y)
        self.line.set_3d_properties(self.trajectory_z)

        # Update start marker (origin - first point)
        self.start.set_data([self.trajectory_x[0]], [self.trajectory_y[0]])
        self.start.set_3d_properties([self.trajectory_z[0]])

        # Update head (end position)
        self.head.set_data([self.trajectory_x[-1]], [self.trajectory_y[-1]])
        self.head.set_3d_properties([self.trajectory_z[-1]])

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the plot."""
        plt.ioff()
        plt.close(self.fig)


def parse_cameras_config(cameras_str: str | None) -> dict[str, Any]:
    """Parse cameras configuration from YAML string."""
    if not cameras_str or cameras_str.strip() == "":
        return {}

    cameras_dict = yaml.safe_load(cameras_str)
    if not isinstance(cameras_dict, dict):
        raise ValueError(f"Cameras config should be a dict, got {type(cameras_dict)}")

    cameras: dict[str, CameraConfig] = {}
    for name, config in cameras_dict.items():
        if not isinstance(config, dict):
            raise ValueError(f"Camera '{name}' config should be a dict, got {type(config)}")

        camera_type = config.pop("type", None)
        if camera_type is None:
            raise ValueError(f"Camera '{name}' missing 'type' field")

        # Map camera type to config class directly
        if camera_type == "opencv":
            camera_config_class = OpenCVCameraConfig
            # Convert index_or_path to string if it's an int (for device indices)
            if "index_or_path" in config and isinstance(config["index_or_path"], int):
                config["index_or_path"] = str(config["index_or_path"])
        elif camera_type == "intelrealsense":
            camera_config_class = RealSenseCameraConfig
            # Keep serial_number_or_name as string - YAML might parse it as int
            if "serial_number_or_name" in config and not isinstance(config["serial_number_or_name"], str):
                # If it's a number, convert to string and zero-pad if it looks like a serial number
                serial = str(config["serial_number_or_name"])
                # Re-add leading zeros if YAML removed them (common with serial numbers starting with 0)
                if len(serial) == 12 and cameras_str.count(f"0{serial}") > 0:
                    serial = "0" + serial
                config["serial_number_or_name"] = serial
        else:
            # Fallback to registry
            camera_config_class = CameraConfig.get_choice_class(camera_type)

        cameras[name] = camera_config_class(**config)

    return cameras


def make_act_pre_post_processors_with_temporal(
    config: ACTConfig,
    dataset_stats: dict | None = None,
    obs_state_horizon: int = 2,
) -> tuple:
    """Create ACT processors with temporal observation handling (UMI-style).

    This MUST match the processors used during training, otherwise
    normalization will be inconsistent and predictions will be wrong.

    Args:
        config: ACT policy configuration
        dataset_stats: Normalization statistics from dataset
        obs_state_horizon: Number of temporal steps in observations

    Returns:
        Tuple of (preprocessor, postprocessor) pipelines
    """
    from lerobot.processor import (
        PolicyProcessorPipeline,
        RenameObservationsProcessorStep,
        TemporalNormalizeProcessor,
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    # Create custom preprocessor with temporal normalization
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        TemporalNormalizeProcessor(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
            obs_state_horizon=obs_state_horizon,
        ),
    ]

    # Standard postprocessor
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    preprocessor = PolicyProcessorPipeline[dict, dict](
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline[dict, dict](
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
    )

    logging.info(f"Created temporal ACT processors with obs_state_horizon={obs_state_horizon}")

    return preprocessor, postprocessor


def build_temporal_image_observation(
    current_img: np.ndarray,
    obs_state_horizon: int,
    image_history: dict | None = None,
    camera_name: str = "camera",
) -> tuple[np.ndarray, dict]:
    """Build temporal image observation matching RelativeEEDataset format.

    For obs_state_horizon > 1, the model expects (T, C, H, W) format.
    This function maintains a history buffer and pads with current frame
    when there is not enough history (matching dataset behavior).

    Args:
        current_img: Current camera image (H, W, C) uint8 [0, 255]
        obs_state_horizon: Number of temporal frames needed
        image_history: Dict of camera_name -> list of previous images (C, H, W) float32
        camera_name: Name of this camera for the history dict

    Returns:
        Tuple of (temporal_images (T, C, H, W) or (C, H, W), updated_image_history)
    """
    if image_history is None:
        image_history = {}

    # Convert to float32 [0, 1] and transpose to (C, H, W)
    img_float = current_img.astype(np.float32) / 255.0
    img_chw = np.transpose(img_float, (2, 0, 1))

    # Initialize history for this camera
    if camera_name not in image_history:
        image_history[camera_name] = []

    # Append current frame
    image_history[camera_name].append(img_chw.copy())

    # Keep only the most recent obs_state_horizon frames
    image_history[camera_name] = image_history[camera_name][-obs_state_horizon:]

    # Pad with current frame if not enough history
    if len(image_history[camera_name]) < obs_state_horizon:
        pad_needed = obs_state_horizon - len(image_history[camera_name])
        image_history[camera_name] = [img_chw.copy()] * pad_needed + image_history[camera_name]

    # Stack into (T, C, H, W) format
    temporal_images = np.stack(image_history[camera_name], axis=0)

    # For obs_state_horizon == 1, squeeze to (C, H, W) to match standard ACT
    if obs_state_horizon == 1:
        temporal_images = temporal_images.squeeze(0)

    return temporal_images, image_history


def load_policy_and_processors(pretrained_path, obs_state_horizon_override=None, device=None):
    """Load ACT policy with optional temporal wrapper and create processors.

    Args:
        pretrained_path: Path to trained model checkpoint directory
        obs_state_horizon_override: Override obs_state_horizon (None = auto-detect)
        device: Torch device

    Returns:
        Tuple of (policy, preprocessor, postprocessor, obs_state_horizon)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = ACTPolicy.from_pretrained(pretrained_path, local_files_only=True)
    policy.eval()
    policy.config.device = str(device)

    policy_obs_state_horizon = getattr(policy.config, 'obs_state_horizon', 1)
    obs_state_horizon = obs_state_horizon_override if obs_state_horizon_override is not None else policy_obs_state_horizon

    if obs_state_horizon > 1:
        original_model = policy.model
        policy.model = TemporalACTWrapper(original_model, policy.config)
        policy.model = policy.model.to(device)
        logger.info(f"Wrapped ACT model with TemporalACTWrapper (obs_state_horizon={obs_state_horizon})")

        wrapper_path = Path(pretrained_path).parent / "temporal_wrapper.pt"
        if not wrapper_path.exists():
            wrapper_path = Path(pretrained_path) / "temporal_wrapper.pt"
        if not wrapper_path.exists():
            wrapper_path = Path(pretrained_path).parent.parent / "temporal_wrapper.pt"
        if wrapper_path.exists():
            wrapper_state_dict = torch.load(wrapper_path, map_location=device)
            model_state = policy.model.state_dict()
            filtered_state = {k: v for k, v in wrapper_state_dict.items() if k in model_state}
            if filtered_state:
                policy.model.load_state_dict(filtered_state, strict=False)
                policy.model = policy.model.to(device)
                logger.info(f"Loaded TemporalACTWrapper parameters from {wrapper_path}")
            else:
                logger.warning(f"No matching parameters in temporal_wrapper.pt, using initialized params")
        else:
            logger.warning(f"No temporal_wrapper.pt found at {wrapper_path}, using initialized params")

        temp_preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy,
            pretrained_path=pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={},
        )

        dataset_stats = None
        for step in temp_preprocessor.steps:
            if hasattr(step, 'stats') and step.stats is not None:
                dataset_stats = step.stats
                break

        if dataset_stats is None:
            raise ValueError("Could not extract normalization stats from pretrained model!")

        preprocessor, postprocessor = make_act_pre_post_processors_with_temporal(
            config=policy.config,
            dataset_stats=dataset_stats,
            obs_state_horizon=obs_state_horizon,
        )
    else:
        logger.info(f"obs_state_horizon=1, TemporalACTWrapper not needed")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy,
            pretrained_path=pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={},
        )

    logger.info("Policy loaded successfully")
    logger.info(f"  Chunk size: {policy.config.chunk_size}")
    logger.info(f"  obs_state_horizon: {obs_state_horizon}")

    return policy, preprocessor, postprocessor, obs_state_horizon


def get_image_from_sample(sample, camera_name="camera"):
    """Extract an HWC uint8 RGB image from a dataset sample.

    Args:
        sample: Dataset sample dict
        camera_name: Camera observation key name

    Returns:
        (H, W, 3) uint8 RGB image
    """
    obs = sample[f"observation.images.{camera_name}"]

    # Handle temporal dimension: (T, C, H, W) -> use last timestep
    if obs.ndim == 4:
        obs = obs[-1]

    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    # (C, H, W) -> (H, W, C)
    if obs.ndim == 3 and obs.shape[0] in [1, 3]:
        obs = obs.transpose(1, 2, 0)

    if obs.dtype != np.uint8:
        obs = (obs * 255).astype(np.uint8)

    if obs.shape[-1] == 1:
        obs = np.repeat(obs, 3, axis=-1)

    return obs


def run_dataset_mode(args):
    """Run dataset mode: project GT or inferred trajectories onto images.

    Without --mp4: display projection and 3D trajectory in real-time.
    With --mp4: save both as MP4 files without displaying.
    """
    dataset_root = args.dataset_root
    dataset_name = Path(dataset_root).name
    repo_id = dataset_name
    camera_name = args.camera_name
    save_mp4 = args.mp4

    # Initialize kinematics offsets
    urdf_path = args.urdf_path
    logger.info(f"Loading kinematic transforms from {urdf_path}")
    T_opt_cam, T_cam_grip = get_kinematic_transforms(urdf_path)

    # Load camera intrinsics
    camera_matrix = load_camera_matrix_from_dataset(dataset_root)

    # Determine obs_state_horizon
    obs_state_horizon = args.obs_state_horizon if args.obs_state_horizon is not None else 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load policy if inference mode
    policy = None
    preprocessor = None
    postprocessor = None
    if args.inference:
        policy, preprocessor, postprocessor, obs_state_horizon = load_policy_and_processors(
            args.pretrained_path, args.obs_state_horizon, device
        )

    # Load dataset
    fps = getattr(policy.config, 'fps', 30) if policy else 30
    chunk_size = policy.config.chunk_size if policy else 100
    action_delta_timestamps = [i / fps for i in range(chunk_size)]
    delta_timestamps = {"action": action_delta_timestamps}

    dataset = RelativeEEDataset(
        repo_id=repo_id,
        root=dataset_root,
        obs_state_horizon=obs_state_horizon,
        delta_timestamps=delta_timestamps,
        compute_stats=False,
    )

    logger.info(f"Dataset loaded: {len(dataset)} frames, {len(dataset.meta.episodes)} episodes")

    mode_label = "inference" if args.inference else "gt"

    if save_mp4:
        output_dir = Path(args.output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create 3D plot figure
    if save_mp4:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure as MplFigure
        fig_3d = MplFigure(figsize=(10, 8))
        FigureCanvasAgg(fig_3d)
        ax_3d = fig_3d.add_subplot(111, projection='3d')
    else:
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        plt.ion()
        fig_3d.show()

    try:
        for ep_idx in args.episode_indices:
            logger.info(f"\nProcessing episode {ep_idx} ({mode_label} mode)...")

            ep_info = dataset.meta.episodes[ep_idx]
            ep_length = ep_info["length"]

            # Find start index
            start_idx = 0
            for i in range(ep_idx):
                start_idx += dataset.meta.episodes[i]["length"]

            logger.info(f"  Episode length: {ep_length} frames")

            if args.inference:
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()

            proj_frames = []
            traj3d_frames = []

            for frame_offset in range(ep_length):
                idx = start_idx + frame_offset
                sample = dataset[idx]

                # Get observation image
                img = get_image_from_sample(sample, camera_name)

                if args.inference:
                    # Build batch from dataset sample
                    obs_state = sample["observation.state"]
                    if isinstance(obs_state, torch.Tensor):
                        obs_state = obs_state.unsqueeze(0).to(device)
                    else:
                        obs_state = torch.from_numpy(obs_state).unsqueeze(0).to(device)

                    batch = {"observation.state": obs_state}

                    # Add image observation
                    obs_img = sample[f"observation.images.{camera_name}"]
                    if isinstance(obs_img, torch.Tensor):
                        obs_img = obs_img.unsqueeze(0).to(device)
                    else:
                        obs_img = torch.from_numpy(obs_img).unsqueeze(0).to(device)
                    batch[f"observation.images.{camera_name}"] = obs_img

                    # Run inference
                    with torch.no_grad():
                        processed_batch = preprocessor(batch)
                        pred_actions = policy.predict_action_chunk(processed_batch)

                        if obs_state_horizon > 1:
                            pred_actions = postprocessor({"action": pred_actions})
                        else:
                            pred_actions = postprocessor(pred_actions)

                        if isinstance(pred_actions, dict) and "action" in pred_actions:
                            pred_actions = pred_actions["action"]
                        actions_np = pred_actions[0].cpu().numpy()  # (chunk_size, 10)
                        
                        trajectory_3d = actions_to_3d_points(actions_np, T_opt_cam, T_cam_grip)
                        points_2d = project_points_to_image(trajectory_3d[1:], camera_matrix)
                        gripper_np = actions_np[:, 9]
                        
                        if args.gt:
                            gt_actions = sample["action"]
                            if isinstance(gt_actions, torch.Tensor):
                                gt_actions = gt_actions.cpu().numpy()
                            trajectory_3d_gt = actions_to_3d_points(gt_actions, T_opt_cam, T_cam_grip)
                            points_2d_gt = project_points_to_image(trajectory_3d_gt[1:], camera_matrix)
                            gripper_gt_np = gt_actions[:, 9]
                        else:
                            gripper_gt_np = None
                else:
                    # Use GT actions from dataset
                    actions_np = sample["action"]
                    if isinstance(actions_np, torch.Tensor):
                        actions_np = actions_np.cpu().numpy()

                    trajectory_3d = actions_to_3d_points(actions_np, T_opt_cam, T_cam_grip)
                    points_2d = project_points_to_image(trajectory_3d[1:], camera_matrix)
                    gripper_np = actions_np[:, 9]
                    gripper_gt_np = None

                # Draw projection on image
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                gripper_to_draw = gripper_np if args.gripper else None
                gripper_gt_to_draw = gripper_gt_np if args.gripper else None
                if args.inference:
                    img_bgr = draw_trajectory_on_image(img_bgr, points_2d, cmap="pred", gripper=gripper_to_draw)
                    if args.gt:
                        img_bgr = draw_trajectory_on_image(img_bgr, points_2d_gt, cmap="gt", gripper=gripper_gt_to_draw)
                else:
                     img_bgr = draw_trajectory_on_image(img_bgr, points_2d, cmap="gt", gripper=gripper_to_draw)

                # Update 3D trajectory plot
                if args.inference and args.gt:
                    update_3d_trajectory_plot(ax_3d, trajectory_3d, frame_offset, ep_idx, mode_label, trajectory_3d_gt=trajectory_3d_gt, gripper=gripper_to_draw, gripper_gt=gripper_gt_to_draw)
                elif args.inference:
                     update_3d_trajectory_plot(ax_3d, trajectory_3d, frame_offset, ep_idx, mode_label, gripper=gripper_to_draw)
                else:
                     update_3d_trajectory_plot(ax_3d, trajectory_3d=trajectory_3d, frame_offset=frame_offset, ep_idx=ep_idx, mode_label=mode_label, gripper=gripper_to_draw)

                if save_mp4:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    proj_frames.append(img_rgb)
                    traj3d_frames.append(render_figure_to_image(fig_3d))
                else:
                    # Display in real-time
                    cv2.imshow(f"Projection ({mode_label})", img_bgr)
                    fig_3d.canvas.draw_idle()
                    fig_3d.canvas.flush_events()
                    key = cv2.waitKey(max(1, int(1000 / fps)))
                    if key == 27:  # ESC to quit
                        logger.info("  ESC pressed, stopping.")
                        return

                if (frame_offset + 1) % 100 == 0:
                    logger.info(f"  Processed {frame_offset + 1}/{ep_length} frames")

            if save_mp4:
                proj_path = output_dir / f"proj_{mode_label}_episode_{ep_idx}.mp4"
                iio.imwrite(proj_path, np.stack(proj_frames), fps=fps)
                logger.info(f"  Saved {proj_path} ({len(proj_frames)} frames)")

                traj3d_path = output_dir / f"traj3d_{mode_label}_episode_{ep_idx}.mp4"
                iio.imwrite(traj3d_path, np.stack(traj3d_frames), fps=fps)
                logger.info(f"  Saved {traj3d_path} ({len(traj3d_frames)} frames)")

    finally:
        if not save_mp4:
            plt.ioff()
            plt.close(fig_3d)
            cv2.destroyAllWindows()

    logger.info("\nDone!")


def run_camera_mode(args):
    """Run camera mode: live inference with physical cameras (original behavior)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize kinematics offsets
    urdf_path = args.urdf_path
    logger.info(f"Loading kinematic transforms from {urdf_path}")
    T_opt_cam, T_cam_grip = get_kinematic_transforms(urdf_path)

    policy, preprocessor, postprocessor, obs_state_horizon = load_policy_and_processors(
        args.pretrained_path, args.obs_state_horizon, device
    )

    # Parse cameras configuration
    cameras_config = parse_cameras_config(args.cameras)
    if not cameras_config:
        raise ValueError("No cameras configured. Please provide --cameras argument.")

    logger.info(f"Configured cameras: {list(cameras_config.keys())}")

    cameras = make_cameras_from_configs(cameras_config)

    for cam_name, camera in cameras.items():
        camera.connect()
        logger.info(f"Connected camera: {cam_name}")

        if hasattr(camera, 'intrinsics'):
            intrinsics = camera.intrinsics
            logger.info(f"  Camera intrinsics ({camera.width}x{camera.height}):")
            logger.info(f"    fx: {intrinsics.fx:.2f}")
            logger.info(f"    fy: {intrinsics.fy:.2f}")
            logger.info(f"    cx: {intrinsics.ppx:.2f}")
            logger.info(f"    cy: {intrinsics.ppy:.2f}")

    plotter = None
    if args.plot_traj:
        plotter = TrajectoryPlotter()
        logger.info("Trajectory plot enabled")

    if args.cameraview:
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow("test", test_img)
            cv2.destroyWindow("test")
            logger.info("Camera view enabled")
        except Exception as e:
            logger.warning(f"Cannot show camera view (headless mode?): {e}")
            logger.warning("Disabling --cameraview")
            args.cameraview = False

    # Prepare identity state observation for relative EE
    identity_pose = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5], dtype=np.float32)

    if obs_state_horizon > 1:
        obs_state = np.stack([identity_pose] * obs_state_horizon, axis=0)
    else:
        obs_state = identity_pose

    logger.info(f"Using identity pose for state observation: {obs_state.shape}")

    # Main Inference Loop
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    image_history = None
    step_count = 0
    trajectory_3d = None
    camera_matrix = None

    logger.info("Starting inference loop...")
    logger.info("Press Ctrl+C to stop\n")

    try:
        while args.num_steps == 0 or step_count < args.num_steps:
            t0 = time.perf_counter()

            batch = {"observation.state": torch.from_numpy(obs_state).unsqueeze(0).to(device)}

            camera_images = {}
            for cam_name, camera in cameras.items():
                img = camera.read()
                camera_images[cam_name] = img

                temporal_img, image_history = build_temporal_image_observation(
                    current_img=img,
                    obs_state_horizon=obs_state_horizon,
                    image_history=image_history,
                    camera_name=cam_name,
                )
                batch[f"observation.images.{cam_name}"] = torch.from_numpy(temporal_img).unsqueeze(0).to(device)

            t_inference = time.perf_counter()
            with torch.no_grad():
                processed_batch = preprocessor(batch)
                pred_actions = policy.predict_action_chunk(processed_batch)

                if obs_state_horizon > 1:
                    pred_actions = postprocessor({"action": pred_actions})
                else:
                    pred_actions = postprocessor(pred_actions)

                if isinstance(pred_actions, dict) and "action" in pred_actions:
                    pred_actions = pred_actions["action"]
                pred_actions = pred_actions[0].cpu().numpy()

            inference_time = time.perf_counter() - t_inference
            print(f"Step {step_count}: Inference {inference_time*1000:.1f}ms | Actions shape: {pred_actions.shape}")

            # Extract gripper states (index 9 in 10D action)
            gripper_states = pred_actions[:, 9] if args.gripper else None

            if plotter is not None:
                trajectory_3d = actions_to_3d_points(pred_actions, T_opt_cam, T_cam_grip)
                plotter.set_trajectory(trajectory_3d)
                plotter.update()

                if camera_matrix is None:
                    for cam in cameras.values():
                        if rs is not None and hasattr(cam, 'rs_pipeline') and cam.rs_pipeline is not None:
                            try:
                                profile = cam.rs_pipeline.get_active_profile()
                                color_stream = profile.get_stream(rs.stream.color)
                                intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                                camera_matrix = np.array([
                                    [intrinsics.fx, 0, intrinsics.ppx],
                                    [0, intrinsics.fy, intrinsics.ppy],
                                    [0, 0, 1]
                                ])
                                break
                            except Exception as e:
                                print(f"Debug: Failed to get intrinsics from RealSense: {e}")
                        elif hasattr(cam, 'intrinsics'):
                            intrinsics = cam.intrinsics
                            camera_matrix = np.array([
                                [intrinsics.fx, 0, intrinsics.ppx],
                                [0, intrinsics.fy, intrinsics.ppy],
                                [0, 0, 1]
                            ])
                            break

            if args.cameraview:
                for cam_name, img in camera_images.items():
                    img_display = img.copy()
                    if img_display.ndim == 3 and img_display.shape[2] == 3:
                        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)

                    if trajectory_3d is not None and camera_matrix is not None:
                        points_2d = project_points_to_image(trajectory_3d[1:], camera_matrix)
                        img_display = draw_trajectory_on_image(img_display, points_2d, gripper=gripper_states)

                    cv2.imshow(f"Camera: {cam_name}", img_display)
                cv2.waitKey(1)

            step_count += 1

            elapsed = time.perf_counter() - t0
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Error during control loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if plotter is not None:
            plotter.close()

        if args.cameraview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        logger.info("Disconnecting cameras...")
        for cam_name, camera in cameras.items():
            camera.disconnect()
        logger.info("Done!")


def main():
    init_logging()

    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize predicted or GT trajectories projected onto camera images"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to trained model checkpoint directory or training output",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Control loop frequency (Hz)",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="urdf/Simulation/SO101/so101_sroi.urdf",
        help="Path to URDF file for calculating True Gripper offsets",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=0,
        help="Number of control steps to run (0 for infinite, camera mode only)",
    )
    parser.add_argument(
        "--obs_state_horizon",
        type=int,
        default=None,
        help="Observation state horizon (auto-detected from policy if not specified)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='Camera configuration in YAML format (enables camera mode)',
    )
    parser.add_argument(
        "--cameraview",
        action="store_true",
        help="Show camera feed in cv2 window (camera mode only)",
    )
    parser.add_argument(
        "--plot-traj",
        action="store_true",
        help="Show real-time 3D trajectory plot (camera mode only)",
    )
    parser.add_argument(
        "--gripper",
        action="store_true",
        help="Visualize gripper state on trajectory",
    )
    # Dataset mode arguments
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Root directory of the dataset (enables dataset mode)",
    )
    parser.add_argument(
        "--episode_indices",
        type=int,
        nargs='+',
        default=None,
        help="Episode indices to process (dataset mode)",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run policy inference in dataset mode (without this, projects GT actions)",
    )
    parser.add_argument(
        "--gt",
        action="store_true",
        help="Overlay GT trajectory on top of predicted trajectory (only works with --inference in dataset mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/debug/animation_relative_ee",
        help="Base output directory for MP4 videos (dataset mode)",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="camera",
        help="Camera observation name in dataset (default: camera)",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Save results as MP4 files instead of displaying (dataset mode)",
    )

    args = parser.parse_args()

    # Dispatch to appropriate mode
    if args.cameras:
        # Camera mode: requires pretrained_path
        if not args.pretrained_path:
            parser.error("--pretrained_path is required for camera mode (when --cameras is provided)")
        run_camera_mode(args)
    elif args.dataset_root:
        # Dataset mode
        if not args.episode_indices:
            parser.error("--episode_indices is required for dataset mode")
        if args.inference and not args.pretrained_path:
            parser.error("--pretrained_path is required when --inference is set")
        run_dataset_mode(args)
    else:
        parser.error("Either --cameras (camera mode) or --dataset_root (dataset mode) must be provided")


if __name__ == "__main__":
    main()
