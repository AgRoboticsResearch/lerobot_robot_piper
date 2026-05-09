"""Move Piper arm to zero position via LeRobot.

Zero position (same as Piper SDK demo):
  j1=0  j2=40.1  j3=-45.8  j4=0  j5=17.2  j6=0
"""

import time
from lerobot_robot_piper import Piper, PiperConfig

# Zero position in degrees (from SDK demo: [0, 0.7, -0.8, 0, 0.3, 0] rad)
ZERO_POS = {
    "joint_1.pos": 0.00,
    "joint_2.pos": 70.00,
    "joint_3.pos": -90.00,
    "joint_4.pos": 0.00,
    "joint_5.pos": 20.00,
    "joint_6.pos": 0.00,
}


def main():
    config = PiperConfig(
        can_interface="can0",
        include_gripper=False,
        use_degrees=True,
        cameras={},
    )
    robot = Piper(config)

    print("=== Go to Zero Position ===")
    print(f"  Target: {ZERO_POS}\n")

    robot.connect()
    print("Connected. Moving to zero...\n")

    robot.send_action(ZERO_POS)

    # Wait and verify
    time.sleep(3.0)
    obs = robot.get_observation()
    print("Current position:")
    for i in range(1, 7):
        key = f"joint_{i}.pos"
        print(f"  {key}: {obs[key]:+8.2f} deg")

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
