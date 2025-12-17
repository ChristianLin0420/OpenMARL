#!/usr/bin/env python3
"""
Template training script for new VLA policies.

This script demonstrates how to set up training using Hydra and the
BaseVLAWorkspace pattern.

Usage:
    # Single GPU
    python train.py task_name=LiftBarrier-rf agent_id=0 data_num=50
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 train.py task_name=LiftBarrier-rf agent_id=0
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

# Register eval resolver for config expressions
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path="new_policy/config",
    config_name="robot_newpolicy",
)
def main(cfg: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print("=" * 80)
        print("Training Configuration")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)
    
    # Instantiate workspace
    workspace = hydra.utils.instantiate(cfg, _recursive_=False)
    
    # Run training
    workspace.run()


if __name__ == "__main__":
    # Handle CUDA_VISIBLE_DEVICES for multi-GPU
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    main()

