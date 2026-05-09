from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig

# Default: calibration.json from the SROI gripper package
_DEFAULT_GRIPPER_CALIBRATION = str(
    Path(__file__).resolve().parent.parent.parent
    / "lerobot_robot_sroi_gripper" / "lerobot_robot_sroi_gripper" / "calibration.json"
)


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    can_interface: str = "can0"
    bitrate: int = 1_000_000
    # Piper SDK returns 6 joints; keep order stable
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i+1}" for i in range(6)])
    # Optional sign flips applied symmetrically to obs/actions (length must match joints)
    joint_signs: list[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    # Allow teleop joints (e.g., SO101) to reference Piper joints directly by name
    joint_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "joint1": "joint1",
            "joint2": "joint2",
            "joint3": "joint3",
            "joint4": "joint4",
            "joint5": "joint5",
            "joint6": "joint6",
        }
    )
    # Expose gripper as "gripper.pos" (0.0 closed - 1.0 open) if True
    include_gripper: bool = False
    # SROI gripper connection
    gripper_port: str = "/dev/ttyACM0"
    gripper_baudrate: int = 921600
    gripper_can_id: int = 0x08
    gripper_recv_id: int = 0x18
    gripper_motor_type: str = "DM4310"
    gripper_kp: float = 10.0   # Impedance stiffness (Nm/rad)
    gripper_kd: float = 1.0    # Impedance damping (Nm·s/rad)
    gripper_calibration_path: str = _DEFAULT_GRIPPER_CALIBRATION
    # Optional cameras; leave empty when not used
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist": OpenCVCameraConfig(
                index_or_path=4,
                width=640,
                height=480,
                fps=30,
                fourcc="MJPG"
            )
        }
    )
    # When False, expose normalized [-100,100] joint percents; when True, degrees
    use_degrees: bool = True
    # Timeout in seconds to wait for SDK EnablePiper during connect
    enable_timeout: float = 5.0