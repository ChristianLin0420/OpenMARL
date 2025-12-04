# OpenVLA Training Pipeline Docker Image
# Base: PyTorch with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

LABEL maintainer="OpenMARL Team"
LABEL description="OpenVLA fine-tuning for RoboFactory multi-agent manipulation"

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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install requirements (cached in image for faster startup)
# These are copied separately so dependency layer is cached
COPY robofactory/requirements.txt /tmp/robofactory_requirements.txt
RUN pip install --no-cache-dir -r /tmp/robofactory_requirements.txt

COPY robofactory/policy/OpenVLA/requirements.txt /tmp/openvla_requirements.txt
RUN pip install --no-cache-dir -r /tmp/openvla_requirements.txt

# Install Flash Attention (optional but recommended)
# Comment out if GPU doesn't support it
RUN pip install --no-cache-dir flash-attn==2.5.5 --no-build-isolation || \
    echo "Warning: Flash Attention installation failed, continuing without it"

# Update numcodecs version
RUN pip install numcodecs==0.12.1

# Copy the full codebase and install in editable mode
COPY . /workspace/OpenMARL/
RUN pip install -e /workspace/OpenMARL

# Create necessary directories
RUN mkdir -p \
    /workspace/OpenMARL/data/rlds_data \
    /workspace/OpenMARL/data/zarr_data \
    /workspace/OpenMARL/data/outputs \
    /workspace/OpenMARL/checkpoints \
    /workspace/OpenMARL/logs

# Add robofactory to PYTHONPATH
ENV PYTHONPATH=/workspace/OpenMARL:/workspace/OpenMARL/robofactory:${PYTHONPATH}

# Set working directory
WORKDIR /workspace

# Expose wandb port (optional)
EXPOSE 8080

# For live file sync, mount your local directory at runtime:
#   -v /path/to/OpenMARL:/workspace/OpenMARL

CMD ["/bin/bash"]

