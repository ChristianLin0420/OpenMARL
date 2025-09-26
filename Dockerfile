# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Use NVIDIA PyTorch container as base image with CUDA 12.2
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Install basic tools and system dependencies
RUN apt-get update && apt-get install -y git tree ffmpeg wget curl
RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so
RUN apt-get install -y libglib2.0-0 libgl1 libglvnd0 libegl1-mesa libgles2-mesa libopengl0 libvulkan1 libmagic1 vulkan-tools

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH="/opt/miniconda3/bin:${PATH}"

# Accept conda Terms of Service
RUN conda config --set always_yes true && \
    conda config --set auto_activate_base false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment with Python 3.9 (as per README)
RUN conda create -n RoboFactory python=3.9 -y

RUN mkdir -p /workspace
WORKDIR /workspace

# Copy only essential files first
COPY setup.py .
COPY robofactory/ ./robofactory/
COPY README.md .

# Copy requirements file
COPY ./robofactory/requirements.txt /requirements.txt

# Activate the environment and install dependencies
RUN /bin/bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate RoboFactory && \
    pip install --no-cache-dir -r /requirements.txt && \
    pip install --upgrade gymnasium==0.29.1 opencv-python==4.11.0.86 && \
    pip install -e . && \
    pip install wandb"

# Set the default command to use the conda environment
CMD ["/bin/bash", "-c", "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate RoboFactory && /bin/bash"]