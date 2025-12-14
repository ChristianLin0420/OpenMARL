# Pi0/Pi0.5 Integration for RoboFactory

This directory contains the complete integration of Physical Intelligence's **Pi0** and **Pi0.5** models into the OpenMARL RoboFactory framework for multi-agent robotic manipulation tasks.

## Overview

Pi0 (π0) and Pi0.5 are flow-based vision-language-action (VLA) models developed by Physical Intelligence. This integration enables fine-tuning these pretrained models on RoboFactory's multi-agent manipulation tasks.

**This integration is fully unified with OpenMARL's training and evaluation interfaces**, meaning you can train and evaluate Pi0/Pi0.5 using the same commands as Diffusion Policy and OpenVLA by simply changing `--policy pi0` or `--policy pi05`.

### Key Features

- ✅ **Unified Interface**: Train/evaluate with `train.sh` and `eval.sh` (same as DP/OpenVLA)
- ✅ **Full Pi0/Pi0.5 Support**: Both base (π0) and improved (π0.5) model variants
- ✅ **Multi-GPU Training**: PyTorch DDP for distributed training across multiple GPUs
- ✅ **LeRobot Data Format**: Automatic conversion from RoboFactory's ZARR format (Step 6)
- ✅ **Wandb Logging**: Comprehensive experiment tracking following openpi patterns
- ✅ **Pretrained Models**: Loads and fine-tunes from openpi's pretrained checkpoints
- ✅ **Action Chunking**: Native support for action horizon prediction
- ✅ **Multi-Camera Support**: Utilizes all 3 camera views (head, global, wrist)
- ✅ **Docker Support**: Reproducible containerized environment

## Architecture

### Model Variants

| Model | Token Length | Flow Matching | Best For |
|-------|--------------|---------------|----------|
| **Pi0** | 48 | No | Faster training, lower memory |
| **Pi0.5** | 200 | Yes | Better performance, more context |

### Camera Mapping

RoboFactory provides 3 cameras per agent, which map to Pi0's expected inputs:

```
RoboFactory ZARR                Pi0 Input Keys
[0] head_camera (side)     →    base_0_rgb
[1] global_camera (overhead) →  left_wrist_0_rgb
[2] wrist_camera (gripper)  →   right_wrist_0_rgb
```

## Installation

### Option 1: Docker (Recommended)

```bash
# Build Docker image
cd OpenMARL/robofactory/policy/Pi0
docker build -t openmarl-pi0 -f Dockerfile ../../..

# Run container with GPU support
docker run --gpus all -it \
  -v $(pwd)/../../..:/workspace/OpenMARL \
  -v ~/.cache:/root/.cache \
  openmarl-pi0
```

### Option 2: Conda/Virtual Environment

```bash
# Activate your environment
conda activate marlvla  # or your preferred environment

# Install RoboFactory base requirements
pip install -r OpenMARL/robofactory/requirements.txt

# Install Pi0-specific requirements
pip install -r OpenMARL/robofactory/policy/Pi0/requirements.txt

# Install openpi as external dependency
pip install git+https://github.com/Physical-Intelligence/openpi.git@main

# Install OpenMARL in editable mode
cd OpenMARL
pip install -e .
```

## Quick Start

### Unified Interface Examples

Pi0 uses the same unified interface as other policies in OpenMARL:

```bash
# Training (same interface for all policies)
bash train.sh --policy pi0    --task LiftBarrier-rf --agent_id 0 --gpus 4  # Pi0
bash train.sh --policy openvla --task LiftBarrier-rf --agent_id 0 --gpus 4  # OpenVLA
bash train.sh --policy dp      --task LiftBarrier-rf --agent_id 0 --gpus 1  # Diffusion Policy

# Evaluation (same interface for all policies)
bash eval.sh --policy pi0    --task LiftBarrier-rf --config robofactory/configs/table/lift_barrier.yaml
bash eval.sh --policy openvla --task LiftBarrier-rf --config robofactory/configs/table/lift_barrier.yaml
bash eval.sh --policy dp      --task LiftBarrier-rf --config robofactory/configs/table/lift_barrier.yaml
```

**Policy Comparison:**

| Policy | Command | Data Format | Cameras | Best For |
|--------|---------|-------------|---------|----------|
| **Pi0** | `--policy pi0` | LeRobot | 3 (fixed) | VLA with language |
| **Pi0.5** | `--policy pi05` | LeRobot | 3 (fixed) | Better VLA performance |
| **OpenVLA** | `--policy openvla` | RLDS | 1 | Vision-language tasks |
| **Diffusion Policy** | `--policy dp` | ZARR | 2-3 | Fast CNN-based policy |

### 1. Prepare Data

The data preparation pipeline automatically includes LeRobot conversion (Step 6) for Pi0/Pi0.5:

```bash
cd OpenMARL/robofactory

# Prepare data for a specific task (includes LeRobot conversion in Step 6)
bash prepare_all_data.sh --num 150 --task LiftBarrier

# Or prepare all tasks
bash prepare_all_data.sh --num 150
```

**Pipeline Steps:**
1. Generate expert demonstrations
2. Convert to H5/PKL formats
3. Convert to ZARR format (for Diffusion Policy)
4. Convert to RLDS format (for OpenVLA)
5. **[Pi0] Convert to LeRobot format** ✅ (Step 6 - NEW!)

**Output:**
```
OpenMARL/robofactory/data/
├── zarr_data/         # For Diffusion Policy
├── rlds_data/         # For OpenVLA
└── lerobot_data/      # For Pi0/Pi0.5 ✅
    ├── LiftBarrier-rf_Agent0_150/
    │   ├── meta/
    │   ├── images/
    │   └── data/
    └── LiftBarrier-rf_Agent1_150/
        └── ...
```

**Verify data:**
```bash
# Check LeRobot datasets were created
ls robofactory/data/lerobot_data/

# Test dataset loading
python << 'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="LiftBarrier-rf_Agent0_150",
    root="robofactory/data/lerobot_data"
)
print(f"✓ Dataset: {len(ds)} frames")
print(f"✓ Keys: {list(ds[0].keys())}")
EOF
```

### 2. Training

#### Option A: Unified Training Interface (Recommended)

Use the unified `train.sh` script from the OpenMARL root:

```bash
cd OpenMARL

# Single GPU Pi0 training
bash train.sh --policy pi0 --task LiftBarrier-rf --agent_id 0

# Multi-GPU Pi0 training (4 GPUs)
bash train.sh --policy pi0 --task LiftBarrier-rf --agent_id 0 --gpus 4

# Pi0.5 training
bash train.sh --policy pi05 --task TwoRobotsStackCube-rf --agent_id 1 --gpus 4

# Train all agents for a task
bash train.sh --policy pi0 --task LiftBarrier-rf --all_agents

# Custom settings
bash train.sh --policy pi0 --task LiftBarrier-rf --agent_id 0 \
  --gpus 4 \
  --seed 42 \
  --batch_size 16 \
  --wandb offline
```

**Unified Training Arguments:**
- `--policy`: Policy type (`pi0` or `pi05`)
- `--task`: Task name (e.g., `LiftBarrier-rf`)
- `--agent_id`: Agent ID (0, 1, 2, ...) or use `--all_agents`
- `--gpus`: Number of GPUs (default: 1)
- `--seed`: Random seed (default: 100)
- `--batch_size`: Batch size per GPU (optional, uses config default)
- `--data_num`: Number of training episodes (default: 160)
- `--wandb`: Wandb mode (`online`, `offline`, `disabled`)
- `--debug`: Enable debug mode

#### Option B: Direct Training Script

For more control, use the training script directly:

```bash
cd OpenMARL/robofactory/policy/Pi0

# Single GPU Training (Pi0)
python train.py \
  --config-name robot_pi0 \
  task.name=LiftBarrier-rf \
  agent_id=0

# Multi-GPU Training (4 GPUs)
torchrun --nproc_per_node=4 train.py \
  --config-name robot_pi0 \
  task.name=LiftBarrier-rf \
  agent_id=0

# Pi0.5 Training
python train.py \
  --config-name robot_pi05 \
  task.name=TwoRobotsStackCube-rf \
  agent_id=1
```

#### Training Arguments

Key configuration overrides:

```bash
# Model settings
model.action_dim=8                    # Action space dimension
model.action_horizon=50               # Action prediction horizon
model.use_gradient_checkpointing=true # Enable for memory efficiency

# Training settings
training.num_epochs=300               # Number of epochs
training.learning_rate=5.0e-4         # Learning rate
training.batch_size=16                # Batch size per GPU
training.gradient_accumulate_every=1  # Gradient accumulation steps

# Logging
logging.wandb_enabled=true            # Enable wandb logging
logging.project=openmarl_pi0          # Wandb project name
logging.log_every_n_steps=20          # Logging frequency
```

### 3. Evaluation

#### Option A: Unified Evaluation Interface (Recommended)

Use the unified `eval.sh` script from the OpenMARL root:

```bash
cd OpenMARL

# Evaluate single task
bash eval.sh --policy pi0 \
  --task LiftBarrier-rf \
  --config robofactory/configs/table/lift_barrier.yaml

# Evaluate Pi0.5
bash eval.sh --policy pi05 \
  --task TwoRobotsStackCube-rf \
  --config robofactory/configs/table/two_robots_stack_cube.yaml

# Evaluate all trained tasks
bash eval.sh --policy pi0 --all_tasks

# Custom evaluation settings
bash eval.sh --policy pi0 \
  --task LiftBarrier-rf \
  --config robofactory/configs/table/lift_barrier.yaml \
  --checkpoint 5000 \
  --num_eval 50 \
  --seed 10000 \
  --data_num 150
```

**Unified Evaluation Arguments:**
- `--policy`: Policy type (`pi0` or `pi05`)
- `--task`: Task name (required for single task)
- `--config`: Config file path (required for single task)
- `--all_tasks`: Evaluate all trained tasks (batch mode)
- `--checkpoint`: Checkpoint step to evaluate (default: 300)
- `--data_num`: Number of training samples used (default: 160)
- `--num_eval`: Number of evaluation episodes (default: 100)
- `--seed`: Starting seed for evaluation (default: 1000)
- `--max_steps`: Maximum steps per episode (default: 250)
- `--debug`: Debug mode (0=quiet, 1=verbose)

#### Option B: Direct Evaluation Script

For more control, use the evaluation script directly:

```bash
cd OpenMARL/robofactory/policy/Pi0

python eval_multi_pi0.py \
  --config configs/table/lift_barrier.yaml \
  --policy_type pi0 \
  --checkpoint_step 5000 \
  --data_num 150 \
  --num_eval_episodes 50 \
  --seed 10000
```

**Direct Script Arguments:**
- `--config`: Path to task config file
- `--policy_type`: `pi0` or `pi05`
- `--checkpoint_step`: Training step of checkpoint to load
- `--data_num`: Number of training samples used
- `--num_eval_episodes`: Number of evaluation episodes
- `--max_steps`: Maximum steps per episode (default: 250)
- `--device`: GPU device (default: cuda:0)

## Directory Structure

```
Pi0/
├── Dockerfile                      # Docker image definition
├── requirements.txt                # Python dependencies
├── train.py                        # Main training script
├── eval_multi_pi0.py              # Multi-agent evaluation script
├── README.md                       # This file
│
└── pi0_policy/                     # Main package
    ├── config/                     # Hydra configs
    │   ├── robot_pi0.yaml         # Pi0 config
    │   ├── robot_pi05.yaml        # Pi0.5 config
    │   └── task/
    │       └── default_task.yaml  # Task config template
    │
    ├── dataset/                    # Data loading
    │   └── robot_lerobot_dataset.py  # LeRobot dataset loader
    │
    ├── model/                      # Model wrappers
    │   └── pi0_wrapper.py         # Pi0 model wrapper
    │
    ├── policy/                     # Policy interface
    │   └── pi0_policy.py          # Policy for evaluation
    │
    ├── utils/                      # Utilities
    │   ├── data_conversion.py     # ZARR → LeRobot conversion
    │   └── task_instructions.py   # Language instructions
    │
    └── workspace/                  # Training logic
        └── pi0_workspace.py       # Training workspace (DDP, logging)
```

## Configuration

### Model Configuration (robot_pi0.yaml)

```yaml
model:
  model_variant: "pi0"                    # "pi0" or "pi05"
  paligemma_variant: "gemma_2b"           # Base vision-language model
  action_expert_variant: "gemma_300m"     # Action prediction head
  action_dim: 8                           # Robot action space
  action_horizon: 50                      # Action chunking horizon
  max_token_len: 48                       # Context length (200 for pi05)
  pretrained_checkpoint: "gs://openpi-assets/checkpoints/pi0_base"
  dtype: "bfloat16"                       # Model precision
  use_gradient_checkpointing: true        # Memory optimization

training:
  num_epochs: 300
  learning_rate: 5.0e-4
  weight_decay: 1.0e-6
  max_grad_norm: 1.0
  val_split: 0.1
  checkpoint_every: 50

dataloader:
  batch_size: 16
  num_workers: 4

logging:
  project: "openmarl_pi0"
  wandb_enabled: true
  log_every_n_steps: 20
```

## Data Format

### Input Format (from LeRobot)

```python
{
  "base_0_rgb": torch.Tensor,        # [B, 3, H, W] - Head camera
  "left_wrist_0_rgb": torch.Tensor,  # [B, 3, H, W] - Global camera
  "right_wrist_0_rgb": torch.Tensor, # [B, 3, H, W] - Wrist camera
  "state": torch.Tensor,             # [B, state_dim] - Joint positions
  "actions": torch.Tensor,           # [B, action_horizon, action_dim]
  "task": List[str],                 # Language instructions
}
```

### Camera Image Processing

- **Input**: HWC uint8 images from ZARR
- **Conversion**: 
  - Transpose to CHW format
  - Normalize to [0, 1] float
  - Handle variable image sizes
- **Output**: Processed tensors for Pi0 model

## Training Tips

### Memory Optimization

For limited GPU memory:

```yaml
model:
  use_gradient_checkpointing: true
  pytorch_training_precision: "bfloat16"

training:
  batch_size: 8                    # Reduce if OOM
  gradient_accumulate_every: 2     # Effective batch = 8 * 2 = 16
```

### Learning Rate Tuning

Default: `5e-4` (from openpi)
- Higher tasks with more data: `1e-3`
- Fine-tuning on small datasets: `1e-4`

### Action Horizon

Default: `50` steps
- Shorter for reactive tasks: `20-30`
- Longer for planning tasks: `50-100`

## Logging and Monitoring

### Wandb Integration

Training automatically logs to wandb:

```python
# Logged metrics (every log_every_n_steps)
- loss: Training loss
- learning_rate: Current LR
- grad_norm: Gradient norm
- val_loss: Validation loss (every val_every epochs)
- checkpoint_step: Checkpoint save events
```

Access logs:
```bash
# View at https://wandb.ai/<your-username>/<project-name>
wandb login  # First time only
```

### Resume Training

Training automatically resumes from the latest checkpoint:

```yaml
training:
  resume: true  # Default: automatically resume if checkpoint exists
```

Checkpoints saved in:
```
data/outputs/YYYY.MM.DD/HH.MM.SS_pi0_<task>_<agent>/
  └── checkpoints/
      ├── wandb_id.txt           # Wandb run ID for resuming
      ├── 5000/
      │   ├── model.safetensors
      │   ├── optimizer.pt
      │   └── metadata.pt
      └── 10000/
          └── ...
```

## Troubleshooting

### Common Issues

#### 1. Import Error: openpi not found

```bash
pip install git+https://github.com/Physical-Intelligence/openpi.git@main
```

#### 2. CUDA Out of Memory

Reduce batch size or enable gradient checkpointing:
```yaml
dataloader.batch_size=8
model.use_gradient_checkpointing=true
```

#### 3. LeRobot Dataset Not Found

Ensure data preparation completed successfully:
```bash
ls data/lerobot_data/
# Should show: <task>_Agent<id>_<num_episodes>/
```

#### 4. Checkpoint Loading Failed

Check checkpoint structure:
```bash
ls data/outputs/.../checkpoints/<step>/
# Should contain: model.safetensors, optimizer.pt, metadata.pt
```

#### 5. Multi-GPU Training Issues

Ensure `CUDA_VISIBLE_DEVICES` and DDP environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py ...
```

## Performance Benchmarks

Expected training time (4x A100 40GB GPUs):

| Task | Episodes | Batch Size | Time/Epoch | Total Time (300 epochs) |
|------|----------|------------|------------|------------------------|
| LiftBarrier-rf | 150 | 64 | ~3 min | ~15 hours |
| TwoRobotsStackCube-rf | 150 | 64 | ~3 min | ~15 hours |

## Citation

If you use this integration, please cite:

```bibtex
@article{black2024pi0,
  title={$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}

@article{openmarl2024,
  title={OpenMARL: Multi-Agent Reinforcement Learning Framework},
  author={OpenMARL Team},
  year={2024}
}
```

## Support

For issues specific to:
- **Pi0 models**: See [openpi repository](https://github.com/Physical-Intelligence/openpi)
- **RoboFactory tasks**: See [OpenMARL documentation](../../README.md)
- **This integration**: Open an issue on the OpenMARL repository

## License

This integration follows the licenses of:
- OpenMARL: MIT License
- openpi: Apache 2.0 License (external dependency)

