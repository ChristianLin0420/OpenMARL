"""
OpenVLA training script for RoboFactory.

Usage:
    Single GPU:
        python train.py --config-name=robot_openvla.yaml
    
    Multi-GPU (torchrun):
        torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --config-name=robot_openvla.yaml
"""

import sys
import os

# CRITICAL: Disable wandb for non-logging ranks IMMEDIATELY before any other imports
# torchrun sets RANK env var; we only want rank 1 to log to wandb
_rank = int(os.environ.get("RANK", 0))
_world_size = int(os.environ.get("WORLD_SIZE", 1))
_is_logging_rank = (_world_size == 1) or (_rank == 1)
if not _is_logging_rank:
    os.environ["WANDB_MODE"] = "disabled"

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'openvla_policy'))

# Use line-buffering for stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
from pathlib import Path

# Register custom resolvers
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / 'openvla_policy' / 'config'),
    config_name='robot_openvla'
)
def main(cfg: OmegaConf):
    """Main training function."""
    # Use the global logging rank flag (set at module level based on RANK env var)
    is_logging_rank = _is_logging_rank
    
    # Resolve config
    OmegaConf.resolve(cfg)
    
    # Print config only from logging rank
    if is_logging_rank:
        print("=" * 80)
        print("Training Configuration:")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)
    
    # Import workspace
    from openvla_policy.workspace.openvla_workspace import OpenVLAWorkspace
    
    # Create workspace
    workspace = OpenVLAWorkspace(cfg)
    
    # Run training
    if is_logging_rank:
        print(f"\nStarting training for task: {cfg.task_name}")
    workspace.run()
    
    if is_logging_rank:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()

