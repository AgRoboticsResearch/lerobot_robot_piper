"""Move Piper arm to home position via LeRobot.

Home position:
  j1=0  j2=0  j3=0  j4=0  j5=25  j6=0
"""

import time
from lerobot_robot_piper import Piper, PiperConfig

HOME_POS = {
    "joint_1.pos": 0.0,
    "joint_2.pos": 0.0,
    "joint_3.pos": 0.0,
    "joint_4.pos": 0.0,
    "joint_5.pos": 25.0,
    "joint_6.pos": 0.0,
}


def main():
    config = PiperConfig(
        can_interface="can0",
        include_gripper=False,
        use_degrees=True,
        cameras={},
    )
    robot = Piper(config)

    print("=== Go to Home Position ===")
    print(f"  Target: {HOME_POS}\n")

    robot.connect()
    print("Connected. Moving to home...\n")

    robot.send_action(HOME_POS)

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
