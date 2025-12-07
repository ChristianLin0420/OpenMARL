"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import os

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

# Add robofactory parent directory to path for simulation evaluation
import pathlib
_script_dir = pathlib.Path(__file__).parent.resolve()
_robofactory_parent = _script_dir.parent.parent.parent  # policy/Diffusion-Policy -> policy -> robofactory -> parent
sys.path.insert(0, str(_robofactory_parent))

import hydra
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()

if __name__ == "__main__":
    main()
