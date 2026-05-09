"""Move arm forward 20cm then back 20cm from initial pose (EE space).

1. Go to initial pose
2. Move EE forward (X+) 200mm
3. Move EE back (X-) 200mm to initial

Usage:
  python test_piper_ik_forward_back.py              # position-only (default)
  python test_piper_ik_forward_back.py --orient     # constrain orientation too
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--orient", action="store_true", help="Constrain orientation (default: position-only)")
    args = parser.parse_args()

    delta_m = 0.2  # 200mm
    orient_w = 0.01 if args.orient else 0.0
    mode_label = "pos+orient" if args.orient else "position-only"

    print(f"=== IK Move: Forward/Back 200mm ({mode_label}) ===\n")

    # Init kinematics
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name=TARGET_FRAME,
        joint_names=JOINT_NAMES,
    )

    # Compute FK at initial
    joints_initial = np.array([0.0, 40.11, -45.84, 0.0, 17.19, 0.0])
    T_initial = kin.forward_kinematics(joints_initial)
    pos_initial = T_initial[:3, 3]
    print(f"  Initial EE: ({pos_initial[0]*1000:.1f}, {pos_initial[1]*1000:.1f}, {pos_initial[2]*1000:.1f}) mm")

    # Compute IK targets
    T_forward = T_initial.copy()
    T_forward[0, 3] += delta_m

    joints_forward = kin.inverse_kinematics(joints_initial, T_forward, position_weight=1.0, orientation_weight=orient_w)
    T_check = kin.forward_kinematics(joints_forward)
    err_forward = np.linalg.norm(T_check[:3, 3] - T_forward[:3, 3]) * 1000

    T_back = T_initial.copy()
    T_back[0, 3] -= delta_m

    joints_back = kin.inverse_kinematics(joints_initial, T_back, position_weight=1.0, orientation_weight=orient_w)
    T_check = kin.forward_kinematics(joints_back)
    err_back = np.linalg.norm(T_check[:3, 3] - T_back[:3, 3]) * 1000

    pos_forward = T_forward[:3, 3]
    pos_back = T_back[:3, 3]
    print(f"  Forward EE: ({pos_forward[0]*1000:.1f}, {pos_forward[1]*1000:.1f}, {pos_forward[2]*1000:.1f}) mm  err={err_forward:.2f}mm")
    print(f"  Back EE:    ({pos_back[0]*1000:.1f}, {pos_back[1]*1000:.1f}, {pos_back[2]*1000:.1f}) mm  err={err_back:.2f}mm")

    if err_forward > 10.0 or err_back > 10.0:
        print("\n  ERROR: IK solution too far off. Aborting.")
        return

    # Connect and move
    print("\n  Connecting to arm...")
    config = PiperConfig(can_interface="can0", include_gripper=False, use_degrees=True, cameras={})
    robot = Piper(config)
    robot.connect()
    print("  Connected.\n")

    # Go to initial
    print("  [1] Going to initial pose...")
    robot.send_action(INITIAL_POS)
    time.sleep(3.0)
    obs = robot.get_observation()
    current = obs_to_joints(obs)
    T_now = kin.forward_kinematics(current)
    print(f"      EE: ({T_now[0,3]*1000:.1f}, {T_now[1,3]*1000:.1f}, {T_now[2,3]*1000:.1f}) mm")

    # Forward 200mm
    print(f"  [2] Moving forward 200mm...")
    robot.send_action(joints_to_action(joints_forward))
    time.sleep(3.0)
    obs = robot.get_observation()
    current = obs_to_joints(obs)
    T_now = kin.forward_kinematics(current)
    print(f"      EE: ({T_now[0,3]*1000:.1f}, {T_now[1,3]*1000:.1f}, {T_now[2,3]*1000:.1f}) mm")

    # Back 200mm
    print(f"  [3] Moving back 200mm...")
    robot.send_action(joints_to_action(joints_back))
    time.sleep(3.0)
    obs = robot.get_observation()
    current = obs_to_joints(obs)
    T_now = kin.forward_kinematics(current)
    print(f"      EE: ({T_now[0,3]*1000:.1f}, {T_now[1,3]*1000:.1f}, {T_now[2,3]*1000:.1f}) mm")

    # Return to initial
    print("  [4] Returning to initial pose...")
    robot.send_action(INITIAL_POS)
    time.sleep(3.0)
    obs = robot.get_observation()
    current = obs_to_joints(obs)
    T_now = kin.forward_kinematics(current)
    print(f"      EE: ({T_now[0,3]*1000:.1f}, {T_now[1,3]*1000:.1f}, {T_now[2,3]*1000:.1f}) mm")

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
