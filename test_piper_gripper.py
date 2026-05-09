"""Test Piper arm + SROI gripper combined."""

import sys
import time

from lerobot_robot_piper import Piper, PiperConfig


def main():
    print("=== Piper Arm + SROI Gripper — Combined Test ===\n")

    config = PiperConfig(
        can_interface="can0",
        include_gripper=True,
        use_degrees=True,
        cameras={},  # no camera for this test
    )
    robot = Piper(config)

    # 1. Connect
    print("[1] Connecting arm + gripper...")
    robot.connect()
    print("    Connected!\n")

    # 2. Read observation
    print("[2] Reading observation...")
    obs = robot.get_observation()
    for key in sorted(obs.keys()):
        if key.startswith("joint") or key.startswith("gripper"):
            print(f"    {key}: {obs[key]}")
    print()

    # 3. Gripper open
    print("[3] Opening gripper (1.0)...")
    robot.send_action({"gripper.pos": 1.0})
    time.sleep(1.5)
    obs = robot.get_observation()
    print(f"    gripper.pos: {obs['gripper.pos']:.4f} (should be ~1.0)\n")

    # 4. Gripper close
    print("[4] Closing gripper (0.0)...")
    robot.send_action({"gripper.pos": 0.0})
    time.sleep(1.5)
    obs = robot.get_observation()
    print(f"    gripper.pos: {obs['gripper.pos']:.4f} (should be ~0.0)\n")

    # 5. Half open
    print("[5] Half-open gripper (0.5)...")
    robot.send_action({"gripper.pos": 0.5})
    time.sleep(1.5)
    obs = robot.get_observation()
    print(f"    gripper.pos: {obs['gripper.pos']:.4f} (should be ~0.5)\n")

    # 6. Stream for 3 seconds
    print("[6] Streaming observation for 3s...")
    start = time.time()
    while time.time() - start < 3.0:
        obs = robot.get_observation()
        joints = "  ".join(f"{obs[f'joint_{i+1}.pos']:+7.1f}" for i in range(6))
        grip = obs.get("gripper.pos", 0.0)
        print(f"    joints: [{joints}]  gripper: {grip:.3f}", flush=True)
        time.sleep(0.1)

    # 7. Disconnect
    print("\n[7] Disconnecting...")
    robot.disconnect()
    print("    Done!")


if __name__ == "__main__":
    main()
