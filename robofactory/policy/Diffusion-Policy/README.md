# Diffusion Policy for RoboFactory

This directory contains the Diffusion Policy implementation for multi-agent robotic manipulation in RoboFactory.

---

## 📋 Overview

Diffusion Policy is a CNN-based diffusion model that learns to predict robot actions from visual observations. It uses a U-Net architecture with diffusion-based action generation for robust manipulation.

### Key Features

- **Architecture**: Conditional U-Net with ResNet vision encoder
- **Action Space**: 8-DoF (7 joint positions + 1 gripper)
- **Data Format**: ZARR datasets
- **Training**: Single GPU
- **Inference**: Real-time action prediction

---

## 🚀 Quick Start

### Training

Use the unified training script from the project root:

```bash
# Train for a specific agent
bash train.sh --policy dp --task LiftBarrier-rf --agent_id 0 --seed 100

# Train all agents for a task
bash train.sh --policy dp --task LiftBarrier-rf --all_agents
```

### Evaluation

```bash
bash eval.sh --policy dp --task LiftBarrier-rf --config configs/table/lift_barrier.yaml
```

---

## 📁 Directory Structure

```
Diffusion-Policy/
├── diffusion_policy/           # Main package
│   ├── common/                 # Utility functions
│   ├── config/                 # Hydra configurations
│   │   └── robot_dp.yaml       # Main training config
│   ├── dataset/                # Dataset loaders
│   │   └── robot_image_dataset.py
│   ├── model/                  # Model architectures
│   │   ├── diffusion/          # Diffusion model components
│   │   ├── vision/             # Vision encoders
│   │   └── common/             # Shared utilities
│   ├── policy/                 # Policy wrappers
│   └── workspace/              # Training workspace
│       └── robotworkspace.py
├── train.py                    # Training entry point
├── eval_dp.py                  # Single agent evaluation
├── eval_multi_dp.py            # Multi-agent evaluation
├── eval_multi.sh               # Evaluation script
└── README.md                   # This file
```

---

## ⚙️ Configuration

Main configuration file: `diffusion_policy/config/robot_dp.yaml`

### Key Parameters

```yaml
# Model
model:
  obs_encoder:
    type: resnet18
    pretrained: true
  
  action_dim: 8
  obs_horizon: 2
  action_horizon: 8
  pred_horizon: 16

# Training
training:
  num_epochs: 300
  learning_rate: 1.0e-4
  batch_size: 64
  seed: 100

# Data
task:
  dataset:
    zarr_path: "data/zarr_data/TaskName_AgentX_150.zarr"
```

---

## 📊 Training Details

### Requirements

- **GPU Memory**: ~8-12 GB
- **Training Time**: ~2-4 hours for 300 epochs
- **Data**: ZARR format dataset

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `obs_horizon` | 2 | Number of observation frames |
| `action_horizon` | 8 | Number of action frames to predict |
| `pred_horizon` | 16 | Total prediction horizon |
| `learning_rate` | 1e-4 | Adam learning rate |
| `batch_size` | 64 | Training batch size |
| `num_epochs` | 300 | Training epochs |

### Training Progress

Training metrics logged to Wandb:
- `train_loss`: Diffusion loss
- `val_loss`: Validation loss (every 10 epochs)
- `learning_rate`: Current learning rate

---

## 🧪 Evaluation

### Multi-Agent Evaluation

Evaluate trained policies in the simulation:

```bash
bash eval.sh --policy dp \
    --task LiftBarrier-rf \
    --config configs/table/lift_barrier.yaml \
    --checkpoint 300 \
    --num_eval 100
```

### Evaluation Metrics

- **Success Rate**: Percentage of successful episodes
- **Episode Length**: Average steps to complete task
- **Motion Planning Rate**: Rate of motion planning failures

---

## 📦 Data Format

Diffusion Policy uses ZARR format datasets:

```
TaskName_AgentX_150.zarr/
├── data/
│   ├── action/          # (N, 8) actions
│   ├── img/             # (N, C, H, W) images
│   └── state/           # (N, D) proprioceptive state
└── meta/
    └── episode_ends/    # Episode boundaries
```

### Data Preparation

```bash
# Generate demonstrations
cd robofactory
python script/generate_data.py --config configs/table/lift_barrier.yaml --num 150

# Convert to ZARR
python script/parse_h5_to_pkl_multi.py --task_name LiftBarrier-rf --load_num 150 --agent_num 2
python script/parse_pkl_to_zarr_dp.py --task_name LiftBarrier-rf --load_num 150 --agent_id 0
```

---

## 🔧 Troubleshooting

### Out of Memory

Reduce batch size:
```bash
bash train.sh --policy dp --task LiftBarrier-rf --agent_id 0 --batch_size 32
```

### Dataset Not Found

Ensure ZARR data exists:
```bash
ls robofactory/data/zarr_data/LiftBarrier-rf_Agent0_150.zarr
```

### Slow Training

- Use GPU with more memory for larger batch sizes
- Enable mixed precision (if supported)

---

## 📚 References

```bibtex
@inproceedings{chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2023}
}
```

---

## 📞 Support

For issues specific to Diffusion Policy in RoboFactory, please check:
1. Dataset format and paths
2. GPU memory availability
3. Training configuration

For general questions, see the [main README](../../../README.md).

