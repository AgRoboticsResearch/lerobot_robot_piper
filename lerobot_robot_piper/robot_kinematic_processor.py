#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("piper_ee_reference_and_delta")
@dataclass
class PiperEEReferenceAndDelta(RobotActionProcessorStep):
    """
    Computes a target end-effector pose from a relative delta command.

    This step takes a desired change in position and orientation (`target_*`) and applies it to a
    reference end-effector pose to calculate an absolute target pose. The reference pose is derived
    from the current robot joint positions using forward kinematics.

    The processor can operate in two modes:
    1.  `use_latched_reference=True`: The reference pose is "latched" or saved at the moment the action
        is first enabled. Subsequent commands are relative to this fixed reference.
    2.  `use_latched_reference=False`: The reference pose is updated to the robot's current pose at
        every step.

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        end_effector_step_sizes: A dictionary scaling the input delta commands.
        motor_names: A list of motor names required for forward kinematics.
        use_latched_reference: If True, latch the reference pose on enable; otherwise, always use the
            current pose as the reference.
        reference_ee_pose: Internal state storing the latched reference pose.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _command_when_disabled: Internal state to hold the last command while disabled.
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    use_latched_reference: bool = (
        True  # If True, latch reference on enable; if False, always use current pose
    )
    use_ik_solution: bool = False

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            q_raw = np.array(
                [
                    float(v)
                    for k, v in observation.items()
                    if isinstance(k, str)
                    and k.endswith(".pos")
                    and k.removesuffix(".pos") in self.motor_names
                ],
                dtype=float,
            )

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Current pose from FK on measured joints
        # RobotKinematics.forward_kinematics expects DEGREES (it converts internally)
        t_curr = self.kinematics.forward_kinematics(q_raw)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))

        # Apply axis remapping from Phone frame to Piper robot frame
        # This corrects the movement direction to match phone orientation
        # Phone Forward (+Y) -> Robot Forward (+X): negate target_x
        # Phone Left (-X) -> Robot Left (+Y): negate target_y
        # Z stays the same
        dx = -tx
        dy = -ty
        dz = tz
        # Rotation remapping to match translation
        twx = wx
        twy = -wy
        twz = -wz

        desired = None

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                # Latched reference mode: latch reference at the rising edge
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = t_curr.copy()
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            delta_p = np.array(
                [
                    dx * self.end_effector_step_sizes["x"],
                    dy * self.end_effector_step_sizes["y"],
                    dz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            r_abs = Rotation.from_rotvec([twx, twy, twz]).as_matrix()
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = ref[:3, :3] @ r_abs
            desired[:3, 3] = ref[:3, 3] + delta_p

            self._command_when_disabled = desired.copy()
        else:
            # While disabled, keep sending the same command to avoid drift.
            if self._command_when_disabled is None:
                # If we've never had an enabled command yet, freeze current FK pose once.
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        # Write action fields
        pos = desired[:3, 3]
        tw = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(tw[0])
        action["ee.wy"] = float(tw[1])
        action["ee.wz"] = float(tw[2])
        action["ee.gripper_vel"] = gripper_vel

        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_ee_pose = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("piper_ee_bounds_and_safety")
@dataclass
class PiperEEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.

    This step ensures that the target end-effector pose remains within a safe operational workspace.
    It also moderates the command to prevent large, sudden movements between consecutive steps.
    Instead of raising exceptions on safety violations, it clamps the values and logs warnings.

    Attributes:
        end_effector_bounds: A dictionary with "min" and "max" keys for position clipping.
        max_ee_step_m: The maximum allowed change in position (in meters) between steps.
        max_ee_rot_step_rad: The maximum allowed change in rotation (in radians) between steps.
        _last_pos: Internal state storing the last commanded position.
        _last_rot: Internal state storing the last commanded rotation (as rotvec).
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    max_ee_rot_step_rad: float = 0.3  # ~17 degrees per step
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_rot: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        rot = np.array([wx, wy, wz], dtype=float)

        # Clip position to workspace bounds
        pos_clipped = np.clip(pos, self.end_effector_bounds["min"], self.end_effector_bounds["max"])
        if not np.allclose(pos, pos_clipped):
            logger.warning(f"EE position clipped: ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) -> ({pos_clipped[0]:.3f},{pos_clipped[1]:.3f},{pos_clipped[2]:.3f})")
            pos = pos_clipped

        # Limit position step size (clamp instead of raise)
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                logger.warning(f"EE position step clamped: {n:.3f}m -> {self.max_ee_step_m}m")
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)

        # Limit rotation step size
        if self._last_rot is not None:
            # Compute rotation difference using rotation composition
            r_last = Rotation.from_rotvec(self._last_rot)
            r_curr = Rotation.from_rotvec(rot)
            r_diff = r_last.inv() * r_curr
            angle_diff = float(np.linalg.norm(r_diff.as_rotvec()))
            
            if angle_diff > self.max_ee_rot_step_rad and angle_diff > 0:
                logger.warning(f"EE rotation step clamped: {np.rad2deg(angle_diff):.1f}° -> {np.rad2deg(self.max_ee_rot_step_rad):.1f}°")
                # Scale down the rotation difference
                r_diff_clamped = Rotation.from_rotvec(r_diff.as_rotvec() * (self.max_ee_rot_step_rad / angle_diff))
                r_new = r_last * r_diff_clamped
                rot = r_new.as_rotvec()

        self._last_pos = pos.copy()
        self._last_rot = rot.copy()

        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(rot[0])
        action["ee.wy"] = float(rot[1])
        action["ee.wz"] = float(rot[2])
        return action

    def reset(self):
        """Resets the last known position and orientation."""
        self._last_pos = None
        self._last_rot = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("piper_inverse_kinematics_ee_to_joints")
@dataclass
class PiperInverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This step translates a Cartesian command (position and orientation of the end-effector) into
    the corresponding joint-space commands for each motor.

    Attributes:
        kinematics: The robot's kinematic model for inverse kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        q_curr: Internal state storing the last joint positions, used as an initial guess for the IK solver.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
            If False, use the solution from the previous step.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Extract joint values in order of motor_names to ensure consistent ordering
        # This is important because IK output will be in the same order
        q_raw = np.array(
            [
                float(observation.get(f"{name}.pos", 0.0))
                for name in self.motor_names
                if name != "gripper"
            ],
            dtype=float,
        )
        if len(q_raw) == 0:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            # RobotKinematics.inverse_kinematics expects DEGREES (it converts internally)
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO: This is sensitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                # RobotKinematics.inverse_kinematics returns DEGREES
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None


@ProcessorStepRegistry.register("piper_gripper_velocity_to_joint")
@dataclass
class PiperGripperVelocityToJoint(RobotActionProcessorStep):
    """
    Converts a gripper velocity command into a target gripper joint position.

    This step integrates a normalized velocity command over time to produce a position command,
    taking the current gripper position as a starting point. It also supports a discrete mode
    where integer actions map to open, close, or no-op.

    Attributes:
        motor_names: A list of motor names, which must include 'gripper'.
        speed_factor: A scaling factor to convert the normalized velocity command to a position change.
        clip_min: The minimum allowed gripper joint position.
        clip_max: The maximum allowed gripper joint position.
        discrete_gripper: If True, treat the input action as discrete (0: open, 1: close, 2: stay).
    """

    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False
    
    # Piper specific: gripper is often handled separately from the motor list 
    # but for consistent interface we often treat it as a joint.
    # We might want to ensure 'gripper' is in the motor list if that's expected.

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        gripper_vel = action.pop("ee.gripper_vel")

        if observation is None:
            raise ValueError("Joints observation is required for computing gripper position")

        # Get current gripper position directly from observation
        current_gripper_pos = float(observation.get("gripper.pos", 0.0))

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: open, 1: close, 2: stay
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_vel = (gripper_vel - 1) * self.clip_max

        # Compute desired gripper position
        delta = gripper_vel * float(self.speed_factor)
        gripper_pos = float(np.clip(current_gripper_pos + delta, self.clip_min, self.clip_max))
        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


@ProcessorStepRegistry.register("piper_joint_safety_clamp")
@dataclass
class PiperJointSafetyClamp(RobotActionProcessorStep):
    """
    Enforces joint-level safety constraints on the robot action.

    This step ensures that:
    1. Joint positions stay within hardware limits
    2. Joint changes between frames are limited to prevent sudden movements
    
    This provides a final safety layer after IK computation, before sending to hardware.

    Attributes:
        motor_names: List of joint motor names to check.
        joint_limits_deg: Dictionary with "min" and "max" lists for joint limits in degrees.
            If None, uses Piper default limits.
        max_joint_step_deg: Maximum allowed joint change per frame in degrees.
        _last_joints: Internal state storing the last commanded joint positions.
    """

    motor_names: list[str]
    joint_limits_deg: dict | None = None
    max_joint_step_deg: float = 10.0  # ~10 degrees per frame at 30fps = ~300 deg/s max
    _last_joints: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Default Piper joint limits (in degrees) from SDK documentation
        if self.joint_limits_deg is None:
            self.joint_limits_deg = {
                "min": [-150.0, 0.0, -170.0, -100.0, -70.0, -120.0],
                "max": [150.0, 180.0, 0.0, 100.0, 70.0, 120.0],
            }

    def action(self, action: RobotAction) -> RobotAction:
        # Extract joint positions from action
        joints = []
        for name in self.motor_names:
            key = f"{name}.pos"
            if key in action:
                joints.append(float(action[key]))
            else:
                # Joint not in action, skip safety check for it
                return action
        
        joints = np.array(joints, dtype=float)
        n_joints = len(joints)
        
        # Ensure limits match number of joints
        min_limits = np.array(self.joint_limits_deg["min"][:n_joints], dtype=float)
        max_limits = np.array(self.joint_limits_deg["max"][:n_joints], dtype=float)

        # 1. Clamp to joint limits
        joints_clamped = np.clip(joints, min_limits, max_limits)
        if not np.allclose(joints, joints_clamped):
            for i, (old, new) in enumerate(zip(joints, joints_clamped)):
                if old != new:
                    logger.warning(f"Joint {self.motor_names[i]} clamped: {old:.1f}° -> {new:.1f}° (limits: [{min_limits[i]:.1f}, {max_limits[i]:.1f}])")
            joints = joints_clamped

        # 2. Limit step size per frame
        if self._last_joints is not None:
            d_joints = joints - self._last_joints
            max_change = float(np.max(np.abs(d_joints)))
            
            if max_change > self.max_joint_step_deg:
                scale = self.max_joint_step_deg / max_change
                joints_limited = self._last_joints + d_joints * scale
                logger.warning(f"Joint step clamped: max change {max_change:.1f}° -> {self.max_joint_step_deg:.1f}°")
                joints = joints_limited

        self._last_joints = joints.copy()

        # Write back to action
        for i, name in enumerate(self.motor_names):
            action[f"{name}.pos"] = float(joints[i])

        return action

    def reset(self):
        """Resets the last known joint positions."""
        self._last_joints = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # This step doesn't change the feature structure
        return features
