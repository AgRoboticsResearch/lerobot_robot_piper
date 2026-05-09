# LeRobot Piper Integration

LeRobot driver for the [Piper 6-DOF robot arm](https://github.com/agilexrobotics/piper_sdk), with end-effector pose control (UMI-EE-pose) via the SmolVLA vision-language-action policy.

## How It Works with LeRobot (UMI-EE-Pose Mode)

The pipeline operates in end-effector (EE) pose space instead of joint space. This means the policy predicts Cartesian deltas (x, y, z, wx, wy, wz) relative to the current EE pose, not individual joint angles.

```
Camera ──► SmolVLA ──► EE-pose deltas (relative)
                              │
                              ▼
                     Absolute EE pose (T_start @ T_delta)
                              │
                              ▼
                     Inverse Kinematics (placo)
                              │
                              ▼
                     Joint safety clamp (limits + velocity)
                              │
                              ▼
                     Piper arm via CAN bus
```

### State & action format (7D)

| Index | Dim | Description |
|-------|-----|-------------|
| 0–2 | x, y, z | EE position in meters |
| 3–5 | wx, wy, wz | EE rotation as axis-angle (rad) |
| 6 | gripper | Normalized position (0 = closed, 1 = open) |

During training, actions are converted to **relative deltas** (pos+rot only; gripper stays absolute). At inference, the postprocessor converts them back to absolute world-frame poses, which are then solved to joint angles via IK.

### Kinematic chain

- 6 revolute joints (`joint1`–`joint6`) with URDF meshes in `lerobot_robot_piper/urdf/`
- `ee_link`: tool center point (116.5mm from `link6` along +Z)
- `camera_link`: wrist-mounted camera frame
- IK solver: [placo](https://github.com/Rhoban/placo) with current joints as initial guess

### Safety layers

1. **IK convergence check** — rejects solutions with >10mm position error
2. **Joint limit enforcement** — hard limits per joint with 0.5° tolerance
3. **Velocity clamping** — max 10° per joint per step
4. **Emergency hold** — repeats current position 3x on failure

## Installation

```bash
# Requires conda env with lerobot + placo
conda activate lerobot_piper_sroi
pip install -e .
```

Dependencies: `lerobot>=0.4.0`, `python-can`, `piper_sdk`, `lerobot-robot-sroi-gripper`

## Usage

### Teleoperation

```bash
lerobot-teleoperate \
    --robot.type=piper \
    --robot.can_interface=can0 \
    --robot.bitrate=1000000 \
    --robot.include_gripper=true \
    --robot.use_degrees=false \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.use_degrees=false
```

### Recording

```bash
lerobot-record \
    --robot.type=piper \
    --robot.can_interface=can0 \
    --robot.bitrate=1000000 \
    --robot.include_gripper=true \
    --robot.use_degrees=false \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
    --dataset.repo_id=local/piper-demo \
    --dataset.root=/path/to/datasets \
    --dataset.single_task="pick and place task" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=45 \
    --dataset.reset_time_s=10 \
    --dataset.video=true
```

### SmolVLA Inference (UMI-EE-Pose)

**Camera mode (live robot execution):**

```bash
python lerobot_robot_piper/run_smolvla_inference.py \
    --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
    --cameras "{color: {type: intelrealsense, serial_number_or_name: '230322274337', width: 640, height: 480, fps: 30}}" \
    --piper can0 \
    --execute \
    --placo_viz \
    --n_execute_steps 8
```

**Dataset mode (dry run, no robot needed):**

```bash
python lerobot_robot_piper/run_smolvla_inference.py \
    --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
    --dataset_root Datasets/sroi_piper_strawberry_picking \
    --max_steps 100
```

**Interactive mode** (press `e` to execute, `r` to re-infer, `q` to quit):

```bash
python lerobot_robot_piper/run_smolvla_inference.py \
    --pretrained_path ... \
    --cameras "..." \
    --piper can0 \
    --placo_viz
```

### ACT Policy Deployment (Joint-Space, Async Inference)

For ACT-based policies that operate in joint space:

```bash
# Terminal 1 — policy server
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 --port=8080 \
    --pretrained_name_or_path=/path/to/model \
    --policy_device=cuda

# Terminal 2 — robot client
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=piper --robot.can_interface=can0 \
    --robot.include_gripper=true --robot.use_degrees=false \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}}' \
    --task="pick green stem" --policy_type=act
```

## Training (UMI-EE-Pose)

Training scripts live in the companion `lerobot/` checkout:

```bash
# Train SmolVLA with EE-pose relative actions
python train_smolvla_umi_ee.py \
    --dataset_repo_id local/piper-strawberry \
    --output_dir outputs/smolvla_umi_ee \
    --training_steps 50000 \
    --save_freq 10000

# Evaluate checkpoint
python eval_smolvla_umi.py \
    --pretrained_path outputs/smolvla_umi_ee/checkpoints/050000/pretrained_model \
    --dataset_repo_id local/piper-strawberry
```

Key training flags (set in `SmolVLAConfig`):
- `derive_state_from_action=True` — state is derived from the action column (2-timestep velocity)
- `use_relative_actions=True` — actions are relative EE-pose deltas
- `relative_exclude_joints=["gripper"]` — gripper stays absolute
- `freeze_vision_encoder=True` — frozen SigLIP backbone
- `train_expert_only=True` — only train the action expert head

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `can_interface` | `can0` | CAN bus interface |
| `bitrate` | `1000000` | CAN bitrate |
| `joint_names` | `joint_1`–`joint_6` | Joint identifiers |
| `joint_signs` | `[1, 1, 1, 1, 1, 1]` | Sign flips for joint directions |
| `joint_aliases` | identity map | Teleoperator-to-Piper joint mapping |
| `include_gripper` | `False` | Enable SROI gripper |
| `gripper_port` | `/dev/ttyACM0` | Gripper serial port |
| `gripper_kp` / `gripper_kd` | `10.0` / `1.0` | Gripper impedance gains |
| `use_degrees` | `True` | Degrees vs normalized [-100,100] |
| `cameras` | OpenCV wrist cam | Camera configurations |
| `enable_timeout` | `5.0` | SDK enable timeout (seconds) |

## Joint Limits

| Joint | Min (deg) | Max (deg) |
|-------|-----------|-----------|
| joint1 | -150 | 150 |
| joint2 | 0 | 180 |
| joint3 | -170 | 0 |
| joint4 | -100 | 100 |
| joint5 | -70 | 70 |
| joint6 | -120 | 120 |

## Test & Debug Scripts

```bash
# Hardware tests
python test_piper_fk_ik.py          # FK/IK roundtrip validation
python test_piper_go_home.py        # Move to home pose
python test_piper_gripper.py        # Gripper control
python test_piper_ik_move.py        # IK-driven EE movement

# Visualization
python live_piper_urdf_viz.py       # Real-time URDF viz (meshcat)
python visualize_trajectory.py      # Replay ORB-SLAM3 trajectory

# SLAM replay
python replay_slam_on_piper.py      # Execute SLAM trajectory on real arm
```
