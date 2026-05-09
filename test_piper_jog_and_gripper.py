"""Piper arm + SROI gripper integration test.

1. Go to initial pose
2. For each joint j1..j6: +10 deg, -10 deg, back to original
3. Gripper: close -> open -> close
"""

import time
from lerobot_robot_piper import Piper, PiperConfig

INITIAL_POS = {
    "joint_1.pos": 0.0,
    "joint_2.pos": 40.11,
    "joint_3.pos": -45.84,
    "joint_4.pos": 0.0,
    "joint_5.pos": 17.19,
    "joint_6.pos": 0.0,
}

JOINT_NAMES = [f"joint_{i}.pos" for i in range(1, 7)]
SWING_DEG = 10.0
STEP_PAUSE = 1.0


def read_all(robot) -> dict:
    obs = robot.get_observation()
    vals = {k: obs.get(k, 0.0) for k in JOINT_NAMES}
    grip = obs.get("gripper.pos", None)
    line = "  ".join(f"j{i+1}={vals[f'joint_{i+1}.pos']:+7.2f}" for i in range(6))
    if grip is not None:
        line += f"  grip={grip:.3f}"
    print(f"    {line}")
    return obs


def move_to(robot, target, label=""):
    if label:
        print(f"    -> {label}")
    robot.send_action(target)
    time.sleep(STEP_PAUSE)


def main():
    config = PiperConfig(
        can_interface="can0",
        include_gripper=True,
        use_degrees=True,
        cameras={},
    )
    robot = Piper(config)

    print("=== Piper Arm + SROI Gripper Integration Test ===\n")
    robot.connect()
    print("Connected.\n")

    # --- Step 1: Go to initial pose ---
    print("[1] Going to initial pose...")
    robot.send_action(INITIAL_POS)
    time.sleep(3.0)
    print("    Arrived:")
    read_all(robot)
    print()

    # --- Step 2: Swing each joint +10, -10, back to original ---
    print(f"[2] Swinging each joint +/-{SWING_DEG} deg (j1 -> j6)...")
    print()

    # Snapshot the base pose after arriving
    obs = robot.get_observation()
    base = {k: obs.get(k, 0.0) for k in JOINT_NAMES}

    for i, joint in enumerate(JOINT_NAMES):
        original = base[joint]
        print(f"  {joint} (original: {original:+.2f})")

        target = dict(base)

        # +10
        target[joint] = original + SWING_DEG
        move_to(robot, target, f"{joint}: {original:+.2f} -> {target[joint]:+.2f}")
        read_all(robot)

        # -10
        target[joint] = original - SWING_DEG
        move_to(robot, target, f"{joint}: {original + SWING_DEG:+.2f} -> {target[joint]:+.2f}")
        read_all(robot)

        # back to original
        target[joint] = original
        move_to(robot, target, f"{joint}: {original - SWING_DEG:+.2f} -> {target[joint]:+.2f}")
        read_all(robot)
        print()

    # --- Step 3: Gripper close -> open -> close ---
    print("[3] Gripper: close -> open -> close...")

    print("    Closing (0.0)...")
    robot.send_action({"gripper.pos": 0.0})
    time.sleep(1.5)
    obs = read_all(robot)
    print(f"    gripper.pos: {obs['gripper.pos']:.4f}")

    print("    Opening (1.0)...")
    robot.send_action({"gripper.pos": 1.0})
    time.sleep(1.5)
    obs = read_all(robot)
    print(f"    gripper.pos: {obs['gripper.pos']:.4f}")

    print("    Keeping open (1.0)...")
    robot.send_action({"gripper.pos": 1.0})
    time.sleep(1.5)
    obs = read_all(robot)
    print(f"    gripper.pos: {obs['gripper.pos']:.4f}")
    print()

    # --- Step 4: Return to initial ---
    print("[4] Returning to initial pose...")
    robot.send_action(INITIAL_POS)
    time.sleep(3.0)
    read_all(robot)

    robot.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
