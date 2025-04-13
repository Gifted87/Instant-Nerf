# Choose a base CUDA image compatible with PyTorch and tiny-cuda-nn
# Check NVIDIA NGC catalog or Docker Hub for appropriate tags
# Example using CUDA 11.8 on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essentials, Python, pip, git, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    wget \
    python3.10 \
    python3-pip \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python/pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /workspace

# Install PyTorch (match CUDA version from base image)
# Check https://pytorch.org/ for the correct command
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install tiny-cuda-nn
# Cloning and building inside the Docker image
RUN apt-get update && apt-get install -y --no-install-recommends ninja-build && rm -rf /var/lib/apt/lists/*
RUN git clone --recursive https://github.com/NVlabs/tiny-cuda-nn /opt/tiny-cuda-nn
WORKDIR /opt/tiny-cuda-nn/bindings/torch
# Ensure CUDA_ARCHITECTURES matches your target GPU(s) or build for common ones
# Example: Building for Ampere (80, 86) and Turing (75)
RUN export TCNN_CUDA_ARCHITECTURES="75;80;86" && python setup.py install

# Copy the rest of the project code
WORKDIR /workspace
COPY . .

# Default command (optional)
# CMD ["python", "src/train.py", "--config", "configs/lego.yaml"]