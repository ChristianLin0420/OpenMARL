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

# Copy setup files first for better caching
COPY setup.py /workspace/
COPY README.md /workspace/
COPY robofactory/__init__.py /workspace/robofactory/

# Install OpenMARL package (following README.md installation steps)
RUN pip install -e .

# Copy and install robofactory requirements
COPY robofactory/requirements.txt /workspace/robofactory/
RUN pip install --no-cache-dir -r robofactory/requirements.txt

# Copy and install OpenVLA-specific requirements
COPY robofactory/policy/OpenVLA/requirements.txt /tmp/openvla_requirements.txt
RUN pip install --no-cache-dir -r /tmp/openvla_requirements.txt

# Install Flash Attention (optional but recommended)
# Comment out if GPU doesn't support it
RUN pip install --no-cache-dir flash-attn==2.5.5 --no-build-isolation || \
    echo "Warning: Flash Attention installation failed, continuing without it"

# Update numcodes version
RUN pip install numcodecs==0.12.1

# Copy the full OpenMARL codebase
COPY . /workspace/

# Create necessary directories
RUN mkdir -p \
    data/rlds_data \
    data/zarr_data \
    data/outputs \
    checkpoints \
    logs

# Add robofactory to PYTHONPATH
ENV PYTHONPATH=/workspace:/workspace/robofactory:${PYTHONPATH}

# Expose wandb port (optional)
EXPOSE 8080

# Default command: launch bash
CMD ["/bin/bash"]