"""Background inference thread for autonomous robot control.

Captures observations, runs model inference, and feeds predicted trajectories
to the controller as schedule_waypoint commands (UMI pattern).

State is read from controller's SharedMemoryRingBuffer (ActualEEPose),
no separate JointBuffer or FK computation needed.
"""

import logging
import threading
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .observation_synchronizer import ObservationSynchronizer

logger = logging.getLogger(__name__)


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
        T[:3, :3] = np.array(
            [
                [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
                [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
                [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
            ]
        )
    return T


def pose_6d_to_matrix(pose_6d: np.ndarray) -> np.ndarray:
    """Convert 6D pose [x, y, z, rx, ry, rz] to 4x4 matrix."""
    T = np.eye(4)
    T[:3, 3] = pose_6d[:3]
    T[:3, :3] = Rotation.from_rotvec(pose_6d[3:]).as_matrix()
    return T


def pose_to_ee_state(T: np.ndarray, gripper: float = 1.0) -> np.ndarray:
    """Convert 4x4 pose matrix to 7D EE state [x, y, z, wx, wy, wz, gripper]."""
    pos = T[:3, 3]
    R = T[:3, :3]
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-10:
        rotvec = np.zeros(3)
    else:
        axis = np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
        axis = axis / (2 * np.sin(angle))
        rotvec = axis * angle
    return np.array([*pos, *rotvec, gripper], dtype=np.float32)


def build_temporal_image(
    img: np.ndarray, history: list | None = None, horizon: int = 1
):
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


class InferenceThread:
    """Background inference thread — feeds controller via schedule_waypoint.

    State comes from controller's SharedMemoryRingBuffer (ActualEEPose),
    matching UMI's pattern where the controller is the single state source.

    Loop:
      1. Check controller queue size < threshold
      2. Capture time-synchronized observation
      3. Build 2-step state tensor from RingBuffer
      4. Check observation similarity (skip if similar)
      5. Run inference (preprocessor -> model -> postprocessor)
      6. Convert predictions to world frame
      7. Send each prediction as controller.schedule_waypoint()
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        controller,
        synchronizer: ObservationSynchronizer,
        task_prompt: str,
        device: str,
        threshold: int = 15,
        fps: float = 30.0,
        similarity_atol: float = 1.0,
    ):
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._controller = controller
        self._synchronizer = synchronizer
        self._task_prompt = task_prompt
        self._device = device
        self._threshold = threshold
        self._fps = fps
        self._similarity_atol = similarity_atol

        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._step_count = 0
        self._error_count = 0
        self._skipped_count = 0
        self._last_state: np.ndarray | None = None
        self._last_infer_ms: float = 0.0
        self._image_history: dict[str, list] = {}

        # Reset pipeline state
        policy.reset()
        for step in preprocessor.steps:
            if hasattr(step, "reset"):
                step.reset()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="inference"
        )
        self._thread.start()
        logger.info("InferenceThread started")

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(
            f"InferenceThread stopped (steps={self._step_count}, "
            f"errors={self._error_count}, skipped={self._skipped_count})"
        )

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> dict:
        return {
            "steps": self._step_count,
            "errors": self._error_count,
            "skipped": self._skipped_count,
            "last_infer_ms": self._last_infer_ms,
            "queue_remaining": self._controller.remaining(),
        }

    def _build_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Build (2, 7) ee_link state from controller RingBuffer.

        Reads ActualEEPose directly from RingBuffer — no FK needed
        on inference side (UMI pattern: controller writes EE pose).
        """
        # Read last 2 states from RingBuffer
        try:
            count = self._controller.ring_buffer.count
            if count >= 2:
                states = self._controller.get_state(k=2)
            else:
                states = self._controller.get_state()
        except Exception:
            states = self._controller.get_state()

        ee_poses = states["ActualEEPose"]
        grippers = states["gripper"]

        if ee_poses.ndim == 1:
            # Single state — duplicate for prev and curr
            ee_now = ee_poses
            ee_prev = ee_poses
            grip_now = float(grippers)
            grip_prev = float(grippers)
        else:
            ee_prev = ee_poses[0]
            ee_now = ee_poses[-1]
            grip_prev = float(grippers[0])
            grip_now = float(grippers[-1])

        T_now = pose_6d_to_matrix(ee_now)
        T_prev = pose_6d_to_matrix(ee_prev)

        T_base = T_now
        state_prev = pose_to_ee_state(
            np.linalg.inv(T_base) @ T_prev, grip_prev
        )
        state_curr = pose_to_ee_state(
            np.linalg.inv(T_base) @ T_now, grip_now
        )
        state_2step = np.stack([state_prev, state_curr])

        return state_2step, T_base

    def _build_batch(
        self, state_2step: np.ndarray, images: dict
    ) -> dict:
        """Build inference batch from state and images."""
        batch = {
            "observation.state": torch.from_numpy(state_2step)
            .unsqueeze(0)
            .to(self._device),
            "task": [self._task_prompt],
        }

        for cam_name, img in images.items():
            temporal_img, hist_list = build_temporal_image(
                img,
                history=self._image_history.get(cam_name),
                horizon=1,
            )
            self._image_history[cam_name] = hist_list
            batch[f"observation.images.{cam_name}"] = (
                torch.from_numpy(temporal_img)
                .unsqueeze(0)
                .to(self._device)
                .float()
            )

        return batch

    def _infer_and_convert(
        self,
        state_2step: np.ndarray,
        images: dict,
        T_base: np.ndarray,
    ) -> np.ndarray:
        """Build batch, run inference, convert to world frame."""
        batch = self._build_batch(state_2step, images)

        with torch.no_grad():
            processed = self._preprocessor(batch)
            pred_actions = self._policy.predict_action_chunk(processed)
            pred_abs = self._postprocessor(pred_actions)

        pred_ee = pred_abs[0]
        if hasattr(pred_ee, "cpu"):
            pred_ee = pred_ee.cpu().numpy()
        elif hasattr(pred_ee, "numpy"):
            pred_ee = pred_ee.numpy()
        else:
            pred_ee = np.asarray(pred_ee)

        pred_world = np.zeros_like(pred_ee)
        for t in range(len(pred_ee)):
            T_delta = action_to_pose(pred_ee[t])
            T_world = T_base @ T_delta
            pred_world[t] = pose_to_ee_state(T_world, pred_ee[t, 6])

        return pred_world

    def _is_similar(self, state_2step: np.ndarray) -> bool:
        if self._similarity_atol <= 0:
            return False
        if self._last_state is None:
            return False
        diff = np.linalg.norm(
            state_2step.flatten() - self._last_state.flatten()
        )
        return diff < self._similarity_atol

    def _loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                # Check queue level
                if self._controller.remaining() >= self._threshold:
                    self._shutdown.wait(0.01)
                    continue

                # Capture observation (camera + state from RingBuffer)
                obs = self._synchronizer.capture()

                # Build state from RingBuffer's ActualEEPose
                state_2step, T_base = self._build_state()

                # Similarity check (skip if similar, unless queue is low)
                force_infer = self._controller.remaining() < 3
                if self._is_similar(state_2step) and not force_infer:
                    self._skipped_count += 1
                    self._shutdown.wait(0.05)
                    continue

                # Inference
                t_infer = time.perf_counter()
                pred_world = self._infer_and_convert(
                    state_2step, obs["images"], T_base
                )
                self._last_infer_ms = (
                    time.perf_counter() - t_infer
                ) * 1000

                # Feed to controller — UMI pattern: exec_actions
                n_sent = self._controller.exec_actions(
                    pred_world,
                    obs_timestamps=obs["t_obs"],
                    dt=1.0 / self._fps,
                )

                self._last_state = state_2step.copy()
                self._step_count += 1

                if self._step_count % 10 == 0:
                    logger.debug(
                        f"Inference step {self._step_count}: "
                        f"{self._last_infer_ms:.0f}ms, "
                        f"sent {n_sent}/{len(pred_world)}, "
                        f"queue={self._controller.remaining()}"
                    )

            except Exception as e:
                self._error_count += 1
                if self._error_count <= 10:
                    logger.warning(f"InferenceThread error: {e}")
                self._shutdown.wait(0.1)
