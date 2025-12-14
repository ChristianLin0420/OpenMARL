"""
Pi0/Pi0.5 training script for RoboFactory.

Follows openpi's train_pytorch.py pattern:
- Hydra for configuration management
- Multi-GPU training with PyTorch DDP
- Wandb logging with run ID tracking
- Checkpoint saving/resuming

Usage:
    # Single GPU training (Pi0)
    python train.py --config-name robot_pi0 task.name=LiftBarrier-rf agent_id=0

    # Multi-GPU training (4 GPUs)
    torchrun --nproc_per_node=4 train.py --config-name robot_pi0 task.name=LiftBarrier-rf agent_id=0

    # Pi0.5 training
    python train.py --config-name robot_pi05 task.name=TwoRobotsStackCube-rf agent_id=1
"""

import os
import sys
from pathlib import Path

# Add pi0_policy to path
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch


@hydra.main(version_base=None, config_path="pi0_policy/config", config_name="robot_pi0")
def main(cfg: DictConfig):
    """
    Main training entry point.
    
    Args:
        cfg: Hydra configuration loaded from YAML
    """
    # Print configuration
    if int(os.environ.get("RANK", 0)) == 0:
        print("="*80)
        print("Pi0 Training Configuration")
        print("="*80)
        print(OmegaConf.to_yaml(cfg))
        print("="*80)
    
    # Instantiate workspace (Hydra will handle _target_ instantiation)
    workspace = hydra.utils.instantiate(cfg, _recursive_=False)
    
    # Run training
    workspace.run()


if __name__ == "__main__":
    # Set CUDA_VISIBLE_DEVICES if specified
    if "CUDA_VISIBLE_DEVICES" not in os.environ and torch.cuda.is_available():
        # Default: use all available GPUs
        num_gpus = torch.cuda.device_count()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    
    main()

