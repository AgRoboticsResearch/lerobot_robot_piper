"""Read Piper arm joint state via LeRobot."""

import time
from lerobot_robot_piper import Piper, PiperConfig


def main():
    config = PiperConfig(
        can_interface="can0",
        include_gripper=False,
        use_degrees=True,
        cameras={},
    )
    robot = Piper(config)

    print("=== Piper Arm Joint State ===\n")
    print("Connecting...")
    robot.connect()

    # Single read
    obs = robot.get_observation()
    print("\n[Joint Positions]")
    for i in range(1, 7):
        key = f"joint_{i}.pos"
        val = obs.get(key, 0.0)
        print(f"  joint_{i}: {val:+8.2f} deg")

    # Continuous stream
    print("\n[Streaming — press Ctrl+C to stop]\n")
    try:
        while True:
            obs = robot.get_observation()
            vals = [f"{obs.get(f'joint_{i+1}.pos', 0.0):+8.2f}" for i in range(6)]
            print(f"  j1={vals[0]}  j2={vals[1]}  j3={vals[2]}  j4={vals[3]}  j5={vals[4]}  j6={vals[5]}", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print("\nDisconnecting...")
    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
