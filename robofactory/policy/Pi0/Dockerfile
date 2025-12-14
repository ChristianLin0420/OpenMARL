# Pi0/Pi0.5 Training Pipeline Docker Image
# Base: PyTorch with CUDA support (same as OpenVLA)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

LABEL maintainer="OpenMARL Team"
LABEL description="Pi0/Pi0.5 fine-tuning for RoboFactory multi-agent manipulation"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Disable tokenizers parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    # Vulkan/EGL dependencies for headless rendering (SAPIEN)
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    libglvnd-dev \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# Create EGL/Vulkan ICD directories for headless rendering
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && mkdir -p /etc/glvnd/egl_vendor.d \
    && mkdir -p /usr/share/egl/egl_external_platform.d \
    && mkdir -p /usr/share/vulkan/icd.d

# Set environment variables for headless rendering
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install base requirements (cached in image for faster startup)
COPY robofactory/requirements.txt /tmp/robofactory_requirements.txt
RUN pip install --no-cache-dir -r /tmp/robofactory_requirements.txt

# Install LeRobot from source (GitHub) with compatible numpy version
# Installing from source ensures all submodules are included
RUN pip install --no-cache-dir 'numpy<2.0,>=1.22.4' && \
    pip install --no-cache-dir git+https://github.com/huggingface/lerobot.git@main && \
    pip install --no-cache-dir --force-reinstall 'numpy<2.0,>=1.22.4'

# Install Pi0-specific dependencies (without lerobot since already installed)
RUN pip install --no-cache-dir \
    'transformers==4.53.2' \
    'einops>=0.7.0' \
    'tyro>=0.7.0' \
    'scipy>=1.9.0'

# Install Flash Attention (optional but recommended for training)
# Comment out if GPU doesn't support it
RUN pip install --no-cache-dir flash-attn==2.5.5 --no-build-isolation || \
    echo "Warning: Flash Attention installation failed, continuing without it"

# Update numcodecs version (for zarr compatibility)
RUN pip install --no-cache-dir numcodecs==0.12.1

# Install JAX/Flax ecosystem (required by openpi imports, even for PyTorch-only usage)
# These are needed because openpi has JAX code mixed with PyTorch code
RUN pip install --no-cache-dir \
    flax \
    jax \
    jaxlib \
    optax \
    orbax-checkpoint \
    tensorflow \
    tensorflow-datasets \
    chex

# Fix numpy version after JAX installation (JAX upgrades it to 2.x)
RUN pip install --no-cache-dir --force-reinstall 'numpy<2.0,>=1.22.4' --no-deps

# Install OpenPI dependencies (required for openpi imports)
RUN pip install --no-cache-dir \
    beartype==0.19.0 \
    sentencepiece \
    jaxtyping==0.2.36 \
    ml-collections \
    equinox \
    augmax \
    numpydantic \
    polars \
    tqdm-loggable \
    pytest

# Fix numpy version again after installing dependencies that upgrade it
RUN pip install --no-cache-dir --force-reinstall 'numpy<2.0,>=1.22.4' --no-deps

# Install openpi package (without dependencies to avoid conflicts)
RUN pip install --no-cache-dir --no-deps git+https://github.com/Physical-Intelligence/openpi.git@main

# Apply transformers patches required by openpi
# OpenPI modifies transformers library to support its custom model architecture
RUN OPENPI_PATH=$(python -c "import openpi; import os; print(os.path.dirname(openpi.__file__))") && \
    TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))") && \
    cp -r $OPENPI_PATH/models_pytorch/transformers_replace/* $TRANSFORMERS_PATH/ && \
    echo "✓ Transformers patches applied from $OPENPI_PATH/models_pytorch/transformers_replace/"

# Final numpy fix to ensure compatibility
RUN pip install --no-cache-dir --force-reinstall 'numpy<2.0,>=1.22.4' --no-deps

# Copy the full codebase and install in editable mode
COPY . /workspace/OpenMARL/
RUN pip install -e /workspace/OpenMARL

# Install wandb with media support
RUN pip install --no-cache-dir wandb[media]

# Install google cloud storage
RUN pip install gcfs

# Create necessary directories
RUN mkdir -p \
    /workspace/OpenMARL/data/lerobot_data \
    /workspace/OpenMARL/data/zarr_data \
    /workspace/OpenMARL/data/rlds_data \
    /workspace/OpenMARL/data/outputs \
    /workspace/OpenMARL/checkpoints \
    /workspace/OpenMARL/logs

# Add robofactory to PYTHONPATH
ENV PYTHONPATH=/workspace/OpenMARL:/workspace/OpenMARL/robofactory:${PYTHONPATH}

# Set working directory to OpenMARL root
WORKDIR /workspace/OpenMARL

# Expose wandb port (optional)
EXPOSE 8080

# Verify critical imports work (all checks must pass)
RUN python -c "import torch; import numpy; print(f'✓ Torch: {torch.__version__}, NumPy: {numpy.__version__}')" && \
    python -c "import sapien; import mani_skill; print('✓ RoboFactory dependencies OK')" && \
    python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('✓ LeRobot OK')" && \
    python -c "import flax; import jax; print('✓ JAX/Flax OK')" && \
    python -c "import openpi; print('✓ OpenPI OK')" && \
    python -c "from openpi.models_pytorch.pi0_pytorch import PI0Pytorch; print('✓ OpenPI PyTorch models OK')" && \
    python -c "from openpi.models_pytorch.pi0_pytorch import PI0Pytorch; from openpi.models.pi0_config import Pi0Config; config = Pi0Config(); print('✓ Transformers patches applied correctly')"

# Note: openpi is now fully installed with transformers patches applied
# All dependencies have been resolved and numpy version is pinned

# Default command
CMD ["/bin/bash"]
