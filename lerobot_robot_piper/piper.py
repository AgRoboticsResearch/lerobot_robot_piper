from typing import Any
import logging

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_piper import PiperConfig
from .piper_sdk_interface import PiperSDKInterface

logger = logging.getLogger(__name__)


class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        # Lazily initialize the SDK interface in connect()
        self._iface: PiperSDKInterface | None = None
        # SROI gripper (when include_gripper=True)
        self._gripper = None  # lerobot_robot_sroi_gripper.motor.Gripper
        self._gripper_closed_rad: float = 0.0
        self._gripper_open_rad: float = 0.0
        self._gripper_range_rad: float = 1.0
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}

    @property
    def is_connected(self) -> bool:
        return (
            self._iface is not None
            and getattr(self._iface, "piper", None) is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.cameras.items()}

    @property
    def observation_features(self) -> dict:
        ft = {**self._motors_ft, **self._cameras_ft}
        if self.config.include_gripper:
            ft["gripper.pos"] = float
        return ft

    @property
    def action_features(self) -> dict:
        ft = {f"{alias}.pos": float for alias in self.config.joint_aliases}
        if self.config.include_gripper:
            ft["gripper.pos"] = float
        return ft

    def connect(self, calibrate: bool = True) -> None:
        # Initialize arm SDK interface
        if self._iface is None:
            self._iface = PiperSDKInterface(
                port=self.config.can_interface,
                enable_timeout=self.config.enable_timeout,
            )
        for cam in self.cameras.values():
            cam.connect()

        # Initialize SROI gripper if enabled
        if self.config.include_gripper:
            self._connect_gripper()

        self.configure()

    def _connect_gripper(self) -> None:
        import json
        from lerobot_robot_sroi_gripper.motor import Gripper, MotorType

        # Load calibration
        cal_path = self.config.gripper_calibration_path
        from pathlib import Path
        path = Path(cal_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Gripper calibration not found: {path}\n"
                "Run calibration first."
            )
        with open(path) as f:
            cal = json.load(f)
        self._gripper_closed_rad = cal["closed_rad"]
        self._gripper_open_rad = cal["open_rad"]
        self._gripper_range_rad = cal["range_rad"]
        logger.info(
            f"Gripper calibration: closed={self._gripper_closed_rad:.4f}, "
            f"open={self._gripper_open_rad:.4f}, range={self._gripper_range_rad:.4f} rad"
        )

        # Map motor type string to enum
        motor_type_map = {"DM4310": MotorType.DM4310}
        motor_type = motor_type_map.get(self.config.gripper_motor_type)
        if motor_type is None:
            raise ValueError(f"Unsupported motor type: {self.config.gripper_motor_type}")

        self._gripper = Gripper(
            port=self.config.gripper_port,
            baudrate=self.config.gripper_baudrate,
            can_id=self.config.gripper_can_id,
            recv_id=self.config.gripper_recv_id,
            motor_type=motor_type,
        )
        self._gripper.connect()
        logger.info("SROI gripper connected")

    def disconnect(self) -> None:
        if self._iface is not None:
            try:
                self._iface.disconnect()
            except Exception:
                pass
            self._iface = None
        if self._gripper is not None:
            try:
                self._gripper.disconnect()
            except Exception:
                pass
            self._gripper = None
            logger.info("SROI gripper disconnected")
        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:  # type: ignore[override]
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _apply_signs(self, joints_deg: list[float]) -> list[float]:
        signs = self.config.joint_signs
        return [d * s for d, s in zip(joints_deg, signs, strict=True)]

    def _get_hw_limits(self) -> tuple[list[float], list[float]]:
        if self._iface is None:
            raise RuntimeError("Piper SDK interface not available")
        min_pos = getattr(self._iface, "min_pos", None)
        max_pos = getattr(self._iface, "max_pos", None)
        if not isinstance(min_pos, list) or not isinstance(max_pos, list) or len(min_pos) < 7 or len(max_pos) < 7:
            raise RuntimeError("Piper SDK limits unavailable")
        return min_pos, max_pos

    def _get_oriented_limits(self) -> tuple[list[float], list[float]]:
        min_pos, max_pos = self._get_hw_limits()
        oriented_min: list[float] = []
        oriented_max: list[float] = []
        for idx, sign in enumerate(self.config.joint_signs):
            hw_min = min_pos[idx]
            hw_max = max_pos[idx]
            if sign >= 0:
                oriented_min.append(hw_min)
                oriented_max.append(hw_max)
            else:
                oriented_min.append(-hw_max)
                oriented_max.append(-hw_min)
        return oriented_min, oriented_max

    def _gripper_raw_to_normalized(self, raw_rad: float) -> float:
        if self._gripper_range_rad == 0:
            return 0.0
        norm = (self._gripper_closed_rad - raw_rad) / self._gripper_range_rad
        return max(0.0, min(1.0, norm))

    def _gripper_normalized_to_raw(self, norm: float) -> float:
        norm = max(0.0, min(1.0, norm))
        return self._gripper_closed_rad - norm * self._gripper_range_rad

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected or self._iface is None:
            raise ConnectionError(f"{self} is not connected.")
        status = self._iface.get_status_deg()

        if not self.config.use_degrees:
            oriented_min, oriented_max = self._get_oriented_limits()

            def deg_to_pct(deg: float, idx: int) -> float:
                rng_min = oriented_min[idx]
                rng_max = oriented_max[idx]
                if rng_max <= rng_min:
                    return 0.0
                pct = (deg - rng_min) / (rng_max - rng_min) * 200.0 - 100.0
                return max(-100.0, min(100.0, pct))
        else:
            def deg_to_pct(deg: float, idx: int) -> float:  # type: ignore[no-redef]
                return deg

        obs = {}
        for i, name in enumerate(self.config.joint_names, start=1):
            deg = status[f"joint_{i}.pos"] * self.config.joint_signs[i - 1]
            obs[f"{name}.pos"] = deg if self.config.use_degrees else deg_to_pct(deg, i - 1)

        # Gripper observation from SROI — wrapped so gripper failures
        # don't kill the joint reads that already succeeded above.
        if self.config.include_gripper and self._gripper is not None:
            try:
                state = self._gripper.send_command(kp=0.0, kd=0.0, position=0.0)
                obs["gripper.pos"] = self._gripper_raw_to_normalized(state.position)
            except Exception as e:
                logger.debug("Gripper read failed: %s", e)

        # Mirror joint values under alias names so teleop processors can access them easily
        for alias, target in self.config.joint_aliases.items():
            target_key = f"{target}.pos"
            alias_key = f"{alias}.pos"
            if target_key in obs and alias_key not in obs:
                obs[alias_key] = obs[target_key]

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected or self._iface is None:
            raise ConnectionError(f"{self} is not connected.")
        # Use current observation as fallback to avoid KeyError / None crash
        try:
            obs = self.get_observation()
        except Exception:
            obs = {f"{name}.pos": 0.0 for name in self.config.joint_names}
            if self.config.include_gripper:
                obs["gripper.pos"] = 0.0

        hw_min, hw_max = self._get_hw_limits()

        if self.config.use_degrees:
            def to_oriented_deg(value: float, idx: int) -> float:  # type: ignore[no-redef]
                return value
        else:
            oriented_min, oriented_max = self._get_oriented_limits()

            def to_oriented_deg(value: float, idx: int) -> float:  # type: ignore[no-redef]
                p = max(-100.0, min(100.0, value))
                p01 = (p + 100.0) / 200.0
                rng_min = oriented_min[idx]
                rng_max = oriented_max[idx]
                if rng_max <= rng_min:
                    return rng_min
                return rng_min + p01 * (rng_max - rng_min)

        name_to_idx = {name: idx for idx, name in enumerate(self.config.joint_names)}
        oriented_deg: dict[str, float] = {}

        for name, idx in name_to_idx.items():
            key = f"{name}.pos"
            raw = action.get(key, obs.get(key, 0.0))
            try:
                val = float(raw)
            except Exception:
                logger.warning("Invalid value for %s: %r, falling back to observation/default", key, raw)
                val = float(obs.get(key, 0.0))
            oriented_deg[name] = to_oriented_deg(val, idx)

        for alias, target in self.config.joint_aliases.items():
            alias_key = f"{alias}.pos"
            if alias_key not in action or target not in oriented_deg:
                continue
            idx = name_to_idx[target]
            raw = action[alias_key]
            try:
                val = float(raw)
            except Exception:
                logger.warning("Invalid value for %s: %r, ignoring alias", alias_key, raw)
                continue
            oriented_deg[target] = to_oriented_deg(val, idx)

        joints_hw_deg = []
        for name, idx in name_to_idx.items():
            deg_oriented = oriented_deg[name]
            deg_hw = deg_oriented * self.config.joint_signs[idx]
            deg_hw = max(hw_min[idx], min(hw_max[idx], deg_hw))
            joints_hw_deg.append(deg_hw)

        # Gripper action via SROI
        gripper_norm = None
        if self.config.include_gripper and self._gripper is not None:
            g_raw = action.get("gripper.pos", obs.get("gripper.pos", 0.5))
            try:
                gripper_norm = max(0.0, min(1.0, float(g_raw)))
            except Exception:
                logger.warning("Invalid gripper.pos value %r, ignoring", g_raw)
                gripper_norm = None

        try:
            # Send arm joints (no Piper built-in gripper)
            self._iface.set_joint_positions_deg(joints_hw_deg, None)
        except Exception as e:
            logger.exception("Failed to send joint positions: %s", e)
            raise

        # Send gripper command separately
        if gripper_norm is not None and self._gripper is not None:
            target_rad = self._gripper_normalized_to_raw(gripper_norm)
            self._gripper.send_command(
                kp=self.config.gripper_kp,
                kd=self.config.gripper_kd,
                position=target_rad,
            )

        return action
