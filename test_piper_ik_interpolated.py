"""Smooth IK move along ee_link Z-axis (forward) over 50 steps (1mm per step)."""

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

N_STEPS = 50
TOTAL_MM = 50.0
STEP_DT = 0.05  # 50ms per step = 20Hz, total ~2.5s


def main():
    total_m = TOTAL_MM / 1000.0
    step_m = total_m / N_STEPS

    print(f"=== Interpolated IK: {TOTAL_MM:.0f}mm along ee_link Z-axis")
    print(f"    {N_STEPS} steps @ {step_m*1000:.1f}mm/step ===\n")

    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="ee_link",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    )

    joints_initial = np.array([0.0, 40.11, -45.84, 0.0, 17.19, 0.0])
    T_initial = kin.forward_kinematics(joints_initial)

    # ee_link Z-axis points forward at initial pose
    forward_dir = T_initial[:3, 2]

    print(f"  Initial ee_link pos: ({T_initial[0,3]*1000:.1f}, "
          f"{T_initial[1,3]*1000:.1f}, {T_initial[2,3]*1000:.1f}) mm")
    print(f"  Z-axis (forward):    ({forward_dir[0]:.3f}, {forward_dir[1]:.3f}, {forward_dir[2]:.3f})")

    # Pre-compute IK for all waypoints
    waypoints = []
    for i in range(N_STEPS + 1):
        T_target = T_initial.copy()
        T_target[:3, 3] += forward_dir * step_m * i

        j_start = waypoints[-1] if waypoints else joints_initial
        j_ik = kin.inverse_kinematics(j_start, T_target, position_weight=1.0, orientation_weight=0.01)
        waypoints.append(j_ik)

        if i % 10 == 0 or i == N_STEPS:
            T_check = kin.forward_kinematics(j_ik)
            dist = step_m * i * 1000
            err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]) * 1000
            print(f"  Waypoint {i:3d}  dist_fwd={dist:.1f}mm  "
                  f"EE=({T_target[0,3]*1000:.1f}, {T_target[1,3]*1000:.1f}, {T_target[2,3]*1000:.1f})  "
                  f"err={err:.2f}mm")

    print(f"\n  {N_STEPS + 1} waypoints computed.\n")

    # Connect
    config = PiperConfig(can_interface="can0", include_gripper=False, use_degrees=True, cameras={})
    robot = Piper(config)
    robot.connect()
    print("Connected.\n")

    def show(label):
        obs = robot.get_observation()
        j = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
        T = kin.forward_kinematics(j)
        print(f"  {label:20s} ee_link: ({T[0,3]*1000:.1f}, {T[1,3]*1000:.1f}, {T[2,3]*1000:.1f}) mm")

    # Go to initial
    print("[1] Going to initial...")
    robot.send_action(INITIAL)
    time.sleep(3.0)
    show("Initial")

    def send_joints(j_ik):
        robot.send_action({f"joint_{i+1}.pos": j_ik[i] for i in range(6)})

    # Forward trajectory (along ee_link Z-axis)
    print(f"[2] Moving forward {TOTAL_MM:.0f}mm along ee_link Z-axis ({N_STEPS} steps)...")
    t_start = time.time()
    for j_ik in waypoints:
        send_joints(j_ik)
        time.sleep(STEP_DT)
    elapsed = time.time() - t_start
    show("After forward")
    print(f"    Elapsed: {elapsed:.2f}s ({N_STEPS/elapsed:.1f} steps/s)")

    # Reverse trajectory
    print(f"[3] Moving back {TOTAL_MM:.0f}mm ({N_STEPS} steps)...")
    t_start = time.time()
    for j_ik in reversed(waypoints):
        send_joints(j_ik)
        time.sleep(STEP_DT)
    elapsed = time.time() - t_start
    show("After back")
    print(f"    Elapsed: {elapsed:.2f}s ({N_STEPS/elapsed:.1f} steps/s)")

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
