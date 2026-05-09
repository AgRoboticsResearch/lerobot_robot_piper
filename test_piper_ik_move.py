"""Test IK on real arm: move EE left/right from initial pose.

1. Connect to arm, go to initial pose
2. Compute FK at initial pose
3. Compute IK for Y-offset targets (left/right in EE space)
4. Send IK joint solutions to arm
5. Return to initial

Usage:
  python test_piper_ik_move.py
  python test_piper_ik_move.py --delta 30     # larger moves (default 20mm)
  python test_piper_ik_move.py --no-dry-run    # actually move the arm
"""

import time
import argparse
import numpy as np

from pathlib import Path
from lerobot.model.kinematics import RobotKinematics
from lerobot_robot_piper import Piper, PiperConfig

URDF_PATH = str(Path(__file__).parent / "lerobot_robot_piper" / "urdf" / "piper_description.urdf")
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
TARGET_FRAME = "link6"

INITIAL_POS = {
    "joint_1.pos": 0.0,
    "joint_2.pos": 40.11,
    "joint_3.pos": -45.84,
    "joint_4.pos": 0.0,
    "joint_5.pos": 17.19,
    "joint_6.pos": 0.0,
}


def obs_to_joints(obs):
    return np.array([obs[f"joint_{i+1}.pos"] for i in range(6)], dtype=float)


def joints_to_action(joints_deg):
    return {f"joint_{i+1}.pos": joints_deg[i] for i in range(6)}


def main():
    parser = argparse.ArgumentParser(description="IK move test: left/right from initial pose")
    parser.add_argument("--delta", type=float, default=20.0, help="EE offset in mm (default: 20)")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually move the arm")
    args = parser.parse_args()

    dry_run = not args.no_dry_run
    delta_m = args.delta / 1000.0  # mm to meters

    print("=== IK Move Test: Left/Right from Initial Pose ===\n")
    print(f"  Delta: ±{args.delta:.0f}mm")
    print(f"  Mode: {'DRY RUN (no arm movement)' if dry_run else 'LIVE'}\n")

    # Init kinematics
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=JOINT_NAMES,
    )

    # Compute FK at initial pose
    joints_initial = np.array([0.0, 40.11, -45.84, 0.0, 17.19, 0.0])
    T_initial = kin.forward_kinematics(joints_initial)
    pos_initial = T_initial[:3, 3]
    print(f"  Initial EE: ({pos_initial[0]*1000:.1f}, {pos_initial[1]*1000:.1f}, {pos_initial[2]*1000:.1f}) mm")

    # Compute IK targets
    moves = [
        ("Right (Y -)", [0, -delta_m, 0]),
        ("Back to center", [0, 0, 0]),
        ("Left (Y +)", [0, delta_m, 0]),
        ("Back to center", [0, 0, 0]),
        ("Forward (X +)", [delta_m, 0, 0]),
        ("Back to center", [0, 0, 0]),
        ("Backward (X -)", [-delta_m, 0, 0]),
        ("Back to center", [0, 0, 0]),
        ("Up (Z +)", [0, 0, delta_m]),
        ("Back to center", [0, 0, 0]),
        ("Down (Z -)", [0, 0, -delta_m]),
        ("Back to center", [0, 0, 0]),
    ]

    print(f"\n  Computing IK for {len(moves)} moves...\n")

    ik_solutions = []
    for name, dx in moves:
        T_target = T_initial.copy()
        T_target[0, 3] += dx[0]
        T_target[1, 3] += dx[1]
        T_target[2, 3] += dx[2]

        joints_ik = kin.inverse_kinematics(
            joints_initial, T_target,
            position_weight=1.0, orientation_weight=0.0,
        )

        T_check = kin.forward_kinematics(joints_ik)
        pos_check = T_check[:3, 3]
        pos_target = T_target[:3, 3]
        err = np.linalg.norm(pos_check - pos_target) * 1000

        ik_solutions.append((name, joints_ik, err))
        status = "OK" if err < 2.0 else ("WARN" if err < 10.0 else "BAD")
        print(f"    {name:20s} err={err:6.2f}mm  joints={[f'{j:+.1f}' for j in joints_ik]}  [{status}]")

    # Check if any solution is BAD
    bad = any(err > 10.0 for _, _, err in ik_solutions)
    if bad:
        print("\n  WARNING: Some IK solutions have large errors. Review before proceeding.")

    if dry_run:
        print("\n  Dry run complete. Use --no-dry-run to move the arm.")
        return

    # Connect to arm and execute
    print("\n  Connecting to arm...")
    config = PiperConfig(can_interface="can0", include_gripper=False, use_degrees=True, cameras={})
    robot = Piper(config)
    robot.connect()
    print("  Connected.\n")

    # Go to initial pose first
    print("  Going to initial pose...")
    robot.send_action(INITIAL_POS)
    time.sleep(3.0)
    obs = robot.get_observation()
    current = obs_to_joints(obs)
    print(f"  Current: {[f'{j:+.1f}' for j in current]}\n")

    # Execute IK moves
    STEP_PAUSE = 2.0
    for name, joints_ik, err in ik_solutions:
        status = "OK" if err < 2.0 else ("WARN" if err < 10.0 else "BAD")
        print(f"  -> {name} [{status}]")
        action = joints_to_action(joints_ik)
        robot.send_action(action)
        time.sleep(STEP_PAUSE)
        obs = robot.get_observation()
        current = obs_to_joints(obs)
        T_actual = kin.forward_kinematics(current)
        pos_actual = T_actual[:3, 3]
        print(f"     Actual EE: ({pos_actual[0]*1000:.1f}, {pos_actual[1]*1000:.1f}, {pos_actual[2]*1000:.1f}) mm")

    # Return to initial
    print("\n  Returning to initial pose...")
    robot.send_action(INITIAL_POS)
    time.sleep(3.0)
    obs = robot.get_observation()
    current = obs_to_joints(obs)
    print(f"  Final: {[f'{j:+.1f}' for j in current]}")

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
