"""Test Piper FK/IK using placo via LeRobot RobotKinematics.

Validates the URDF model against the real arm.
Uses link6 as the end-effector frame (no gripper).

Usage:
  python test_piper_fk_ik.py              # FK only (no hardware)
  python test_piper_fk_ik.py --live       # Compare FK with real arm readings
  python test_piper_fk_ik.py --ik         # Test IK round-trip
"""

import sys
import time
import argparse
import numpy as np

from lerobot.model.kinematics import RobotKinematics

from pathlib import Path
URDF_PATH = str(Path(__file__).parent / "lerobot_robot_piper" / "urdf" / "piper_description.urdf")
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
TARGET_FRAME = "link6"


def test_fk(kin):
    """Test FK at several known joint configurations."""
    print("=== FK Test ===\n")

    configs = {
        "Zero":       [0,    0,     0,    0,    0,    0],
        "Home":       [0,    0,     0,    0,   25,    0],
        "Initial":    [0, 40.11, -45.84,  0, 17.19,  0],
        "Reach out":  [0,   90,     0,    0,    0,    0],
        "Folded":     [0,    0,   -90,    0,    0,    0],
        "J1 left":  [-45, 40.11, -45.84,  0, 17.19,  0],
        "J1 right": [ 45, 40.11, -45.84,  0, 17.19,  0],
    }

    for name, joints in configs.items():
        T = kin.forward_kinematics(np.array(joints, dtype=float))
        pos = T[:3, 3]
        print(f"  {name:12s} joints={joints}")
        print(f"  {'':12s} EE pos = ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")
        print()


def test_fk_live(kin):
    """Read real arm joints, compute FK, display EE position."""
    print("=== FK Live Test (comparing with real arm) ===\n")

    from lerobot_robot_piper import Piper, PiperConfig

    config = PiperConfig(can_interface="can0", include_gripper=False, use_degrees=True, cameras={})
    robot = Piper(config)
    robot.connect()
    print("Arm connected.\n")

    print("Reading joints and computing FK (Ctrl+C to stop)...\n")
    try:
        while True:
            obs = robot.get_observation()
            joints = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)], dtype=float)
            T = kin.forward_kinematics(joints)
            pos = T[:3, 3]
            jstr = " ".join(f"{j:+7.2f}" for j in joints)
            print(
                f"  joints: [{jstr}]  "
                f"EE: ({pos[0]*1000:7.1f}, {pos[1]*1000:7.1f}, {pos[2]*1000:7.1f}) mm",
                flush=True,
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass

    robot.disconnect()
    print("\nArm disconnected.")


def test_ik_roundtrip(kin):
    """Test IK: compute FK → use result as IK target → verify round-trip."""
    print("=== IK Round-Trip Test ===\n")

    configs = {
        "Zero":     [0,    0,     0,    0,    0,    0],
        "Initial":  [0, 40.11, -45.84,  0, 17.19,  0],
        "Reach":    [0,   90,     0,    0,    0,    0],
        "Tuck":     [0,   45,   -90,    0,   45,    0],
    }

    for name, joints_start in configs.items():
        joints_start = np.array(joints_start, dtype=float)

        # FK: joints → EE pose
        T_target = kin.forward_kinematics(joints_start)
        pos_target = T_target[:3, 3]

        # IK: EE pose → joints (using start as initial guess)
        joints_ik = kin.inverse_kinematics(joints_start, T_target)
        pos_ik = kin.forward_kinematics(joints_ik)[:3, 3]

        # Compare
        joint_err = np.max(np.abs(joints_ik - joints_start))
        pos_err = np.linalg.norm(pos_ik - pos_target) * 1000  # mm

        status = "OK" if pos_err < 1.0 else "DRIFT"
        print(f"  {name:10s} pos_err={pos_err:6.2f}mm  joint_err={joint_err:6.2f}deg  [{status}]")

    print()


def test_ik_targets(kin):
    """Test IK with specific EE targets."""
    print("=== IK Target Test ===\n")

    joints_start = np.array([0, 40.11, -45.84, 0, 17.19, 0], dtype=float)
    T_start = kin.forward_kinematics(joints_start)
    pos_start = T_start[:3, 3]
    print(f"  Start EE: ({pos_start[0]*1000:.1f}, {pos_start[1]*1000:.1f}, {pos_start[2]*1000:.1f}) mm\n")

    deltas = [
        ("X +20mm", [0.02, 0, 0]),
        ("X -20mm", [-0.02, 0, 0]),
        ("Y +20mm", [0, 0.02, 0]),
        ("Z +20mm", [0, 0, 0.02]),
        ("Z -20mm", [0, 0, -0.02]),
    ]

    for label, orient_weight in [("pos+orient (default)", 0.01), ("position-only", 0.0)]:
        print(f"  --- {label} (orientation_weight={orient_weight}) ---")
        for name, dx in deltas:
            T_target = T_start.copy()
            T_target[0, 3] += dx[0]
            T_target[1, 3] += dx[1]
            T_target[2, 3] += dx[2]

            joints_ik = kin.inverse_kinematics(
                joints_start, T_target,
                position_weight=1.0, orientation_weight=orient_weight,
            )
            T_result = kin.forward_kinematics(joints_ik)
            pos_result = T_result[:3, 3]
            pos_target = T_target[:3, 3]
            err = np.linalg.norm(pos_result - pos_target) * 1000
            status = "OK" if err < 2.0 else ("WARN" if err < 10.0 else "BAD")
            print(f"    {name:8s} err={err:6.2f}mm  joints={[f'{j:+.1f}' for j in joints_ik]}  [{status}]")
        print()


def main():
    parser = argparse.ArgumentParser(description="Piper FK/IK test")
    parser.add_argument("--live", action="store_true", help="Compare FK with real arm")
    parser.add_argument("--ik", action="store_true", help="Run IK tests")
    args = parser.parse_args()

    print(f"URDF: {URDF_PATH}")
    print(f"Target frame: {TARGET_FRAME}")
    print(f"Joints: {JOINT_NAMES}\n")

    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=JOINT_NAMES,
    )
    print("RobotKinematics initialized.\n")

    # Always run FK
    test_fk(kin)

    # IK tests
    if args.ik:
        test_ik_roundtrip(kin)
        test_ik_targets(kin)

    # Live comparison
    if args.live:
        test_fk_live(kin)


if __name__ == "__main__":
    main()
