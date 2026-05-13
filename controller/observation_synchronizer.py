"""Observation synchronizer: align camera frames with robot state from controller RingBuffer.

Camera read is the reference clock. After reading the camera, robot state
is read from the controller's SharedMemoryRingBuffer (UMI pattern).
"""

import time


class ObservationSynchronizer:
    """Aligns camera frames and robot state to a common timestamp.

    Reads latest state from controller's SharedMemoryRingBuffer.
    No separate JointBuffer or JointReader needed (UMI: single state source).

    Usage:
        sync = ObservationSynchronizer(controller, cameras)
        obs = sync.capture()
        # obs = {"images": {...}, "joints": (6,), "gripper": float, "t_obs": float}
    """

    def __init__(self, controller, cameras: dict):
        self.controller = controller
        self.cameras = cameras

    def capture(self) -> dict:
        """Capture a time-synchronized observation.

        Steps:
          1. Record t_before
          2. Read camera(s) — dominant latency (~33ms at 30fps)
          3. Record t_after
          4. t_camera = midpoint — best estimate of shutter time
          5. Read latest state from controller RingBuffer

        Returns dict with keys: images, joints, gripper, t_obs.
        """
        t_before = time.monotonic()

        images = {}
        for cam_name, camera in self.cameras.items():
            images[cam_name] = camera.read()

        t_after = time.monotonic()
        t_camera = (t_before + t_after) / 2.0

        # Read latest state from controller RingBuffer (UMI pattern)
        state = self.controller.get_state()

        return {
            "images": images,
            "joints": state["ActualJointState"],
            "gripper": float(state["gripper"]),
            "t_obs": t_camera,
        }
