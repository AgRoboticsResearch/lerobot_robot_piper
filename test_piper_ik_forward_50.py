"""Forward 50mm from initial pose with orientation constraint."""

import time
import numpy as np
from pathlib import Path
from lerobot.model.kinematics import RobotKinematics
from lerobot_robot_piper import Piper, PiperConfig

URDF_PATH = str(Path(__file__).parent / "lerobot_robot_piper" / "urdf" / "piper_description.urdf")

INITIAL = {
    "joint_1.pos": 0.0,
    "joint_2.pos": 40.11,
    "joint_3.pos": -45.84,
    "joint_4.pos": 0.0,
    "joint_5.pos": 17.19,
    "joint_6.pos": 0.0,
}


def main():
    print("=== Forward 50mm (pos+orient) ===\n")

    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="link6",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    )

    joints_initial = np.array([0.0, 40.11, -45.84, 0.0, 17.19, 0.0])
    T_initial = kin.forward_kinematics(joints_initial)

    # Forward 50mm with orientation constraint
    T_fwd = T_initial.copy()
    T_fwd[0, 3] += 0.05
    joints_fwd = kin.inverse_kinematics(
        joints_initial, T_fwd, position_weight=1.0, orientation_weight=0.01,
    )
    T_check = kin.forward_kinematics(joints_fwd)
    err = np.linalg.norm(T_check[:3, 3] - T_fwd[:3, 3]) * 1000
    print(f"  IK err: {err:.2f}mm  joints={[f'{j:+.1f}' for j in joints_fwd]}\n")

    if err > 10.0:
        print("  ERROR: IK off. Aborting.")
        return

    config = PiperConfig(can_interface="can0", include_gripper=False, use_degrees=True, cameras={})
    robot = Piper(config)
    robot.connect()
    print("Connected.\n")

    def show(label):
        obs = robot.get_observation()
        j = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
        T = kin.forward_kinematics(j)
        print(f"  {label:20s} EE: ({T[0,3]*1000:.1f}, {T[1,3]*1000:.1f}, {T[2,3]*1000:.1f}) mm")

    print("[1] Going to initial...")
    robot.send_action(INITIAL)
    time.sleep(3.0)
    show("Initial")

    print("[2] Moving forward 50mm...")
    robot.send_action({f"joint_{i+1}.pos": joints_fwd[i] for i in range(6)})
    time.sleep(3.0)
    show("Forward +50mm")

    print("[3] Back to initial...")
    robot.send_action(INITIAL)
    time.sleep(3.0)
    show("Back to initial")

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
