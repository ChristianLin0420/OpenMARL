<div align="center">
<h1>OpenMARL: Open Multi-Agent Reinforcement Learning</h1>

<a href="https://arxiv.org/abs/2503.16408"><img src="https://img.shields.io/badge/arxiv-2503.16408-b31b1b" alt="arXiv"></a>
<a href="https://iranqin.github.io/robofactory/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/datasets/FACEONG/RoboFactory_Dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue'></a>
</div>

## 🧠 Overview

OpenMARL is an open-source framework for multi-agent reinforcement learning in embodied AI. It features **RoboFactory**, a benchmark for embodied multi-agent manipulation based on [ManiSkill](https://www.maniskill.ai/). Leveraging compositional constraints and specifically designed interfaces, OpenMARL provides an automated data collection framework for embodied multi-agent systems.

<div align="center">
  <img src="./images/motivation.png" width="950"/>
</div>

## 🚀 Quick Start

First, clone this repository to your local machine, and install [vulkan](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) and the following dependencies.

```bash
git clone https://github.com/ChristianLin0420/OpenMARL.git
conda create -n openmarl python=3.9
conda activate openmarl
cd OpenMARL
pip install -e .
cd robofactory
pip install -r requirements.txt
# (optional): conda install -c conda-forge networkx=2.5
```

Then download the 3D assets for RoboFactory tasks:

```bash
python script/download_assets.py 
```

For complex scenes like [RoboCasa](https://github.com/robocasa/robocasa), download them using the following command. Note that if you use these scenes in your work please cite the scene dataset authors.

```bash
python -m mani_skill.utils.download_asset RoboCasa
```

Now, try to run a task with just a line of code:

```bash
# Table scene
python script/run_task.py configs/table/lift_barrier.yaml

# RoboCasa scene (after downloading RoboCasa assets)
python script/run_task.py configs/robocasa/lift_barrier.yaml
```

### 🛠 Installing OpenGL/EGL Dependencies on Headless Debian Servers

If you are running simulation environments on a **headless Debian server** without a graphical desktop, you will need to install a minimal set of OpenGL and EGL libraries to ensure compatibility.

Run the following commands to install the necessary runtime libraries:

```bash
sudo apt update
sudo apt install libgl1 libglvnd0 libegl1-mesa libgles2-mesa libopengl0
```

## 📦 Data Download & Processing

### Option 1: Automated Data Preparation (Recommended)

We provide an automated script that handles all data generation and processing steps for all tasks:

```bash
bash prepare_all_data.sh
```

This script will:
1. Download required assets
2. Generate demonstration data for all tasks (150 episodes each)
3. Convert data from H5 → PKL → ZARR format automatically
4. Skip already prepared tasks to save time

The script processes both `table` and `robocasa` scene types for all available tasks:
- CameraAlignment-rf
- LiftBarrier-rf
- LongPipelineDelivery-rf
- PassShoe-rf
- PickMeat-rf
- PlaceFood-rf
- StackCube-rf
- StrikeCube-rf
- TakePhoto-rf
- ThreeRobotsStackCube-rf
- TwoRobotsStackCube-rf

**Note:** The data preparation process can take several hours depending on your hardware.

### Option 2: Manual Data Generation (For Individual Tasks)

If you want to generate data for a specific task:

```bash
# Format: python script/generate_data.py --config {config_path} --num {traj_num} [--save-video]
python script/generate_data.py --config configs/table/lift_barrier.yaml --num 150 --save-video
```

The generated demonstration data will be saved in the `demos/` folder.

## 🧪 Train & Evaluate Policy

### Data Processing

If you used the automated preparation script (`prepare_all_data.sh`), your data is already in ZARR format and ready for training. Skip to the [Train](#train) section.

If you generated data manually, you need to convert it from H5 to ZARR format:

```bash
# 1. Create data directories (first time only)
mkdir -p data/{h5_data,pkl_data,zarr_data}

# 2. Move your H5 and JSON files to data/h5_data/
mv demos/{task_name}/motionplanning/*.h5 data/h5_data/{task_name}.h5
mv demos/{task_name}/motionplanning/*.json data/h5_data/{task_name}.json

# 3. Convert H5 → PKL (handles both single and multi-agent tasks)
# The script automatically detects the number of agents
python script/parse_h5_to_pkl_multi.py --task_name LiftBarrier-rf --load_num 150 --agent_num 2

# 4. Convert PKL → ZARR (for each agent)
# For a 2-agent task like LiftBarrier-rf:
python script/parse_pkl_to_zarr_dp.py --task_name LiftBarrier-rf --load_num 150 --agent_id 0
python script/parse_pkl_to_zarr_dp.py --task_name LiftBarrier-rf --load_num 150 --agent_id 1
```

**Note:** The `--agent_num` parameter should match the number of robots in your task configuration.

### Train

We currently provide training code for [Diffusion Policy](https://arxiv.org/pdf/2303.04137) (DP), and we plan to provide more policies in the future.
You can train the DP model through the following code:

```bash
bash policy/Diffusion-Policy/train.sh ${task_name} ${load_num} ${agent_id} ${seed} ${gpu_id}
# Example:
bash policy/Diffusion-Policy/train.sh LiftBarrier-rf 150 0 100 0
bash policy/Diffusion-Policy/train.sh LiftBarrier-rf 150 1 100 0
```

### Evaluation

Use the .ckpt file to evaluate your model results after the training is completed. When setting DEBUG_MODE to 1, it will open the visual window and output more info.

```bash
bash policy/Diffusion-Policy/eval_multi.sh ${config_name} ${DATA_NUM} ${CHECKPOINT_NUM} ${DEBUG_MODE} ${TASK_NAME}
# Example
bash policy/Diffusion-Policy/eval_multi.sh configs/table/lift_barrier.yaml 150 300 1 LiftBarrier-rf
```

## 🔗 Community & Contact

For any questions or research collaboration opportunities, please don't hesitate to reach out：yiranqin@link.cuhk.edu.cn, faceong02@gmail.com, akikaze@sjtu.edu.cn.

## 📚 BibTeX

```bibtex
@article{qin2025robofactory,
  title={RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints},
  author={Qin, Yiran and Kang, Li and Song, Xiufeng and Yin, Zhenfei and Liu, Xiaohong and Liu, Xihui and Zhang, Ruimao and Bai, Lei},
  journal={arXiv preprint arXiv:2503.16408},
  year={2025}
}
```
