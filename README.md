
# Instant-NeRF: Real-Time 3D Reconstruction with Hash Grid Encoding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Add a LICENSE file -->

This repository implements Neural Radiance Fields (NeRF) accelerated using hash grid encoding techniques, inspired by NVIDIA's Instant-NGP paper. The goal is to achieve significantly faster training and real-time rendering for 3D scene reconstruction from 2D images.

This implementation includes features like automatic mixed precision (AMP), hierarchical sampling, and **graceful fallback** to a standard MLP encoder if the `tiny-cuda-nn` library is unavailable.

## Key Features

*   **GPU-Accelerated Hash Grid:** Leverages `tiny-cuda-nn` for efficient multi-resolution hash encoding on the GPU (if available).
*   **Graceful Fallback:** Automatically switches to a PyTorch MLP encoder if `tiny-cuda-nn` cannot be imported or initialized, allowing the code to run (albeit slower) without the specialized library.
*   **Fast Convergence:** Aims for high-quality results significantly faster (targeting 100x speedup with TCNN backend) than vanilla NeRF implementations.
*   **Differentiable Volume Rendering:** Implements the core NeRF rendering pipeline with hierarchical sampling and chunked processing for memory efficiency.
*   **Configurable:** Uses YAML files (`configargparse`) for setting parameters, allowing easy overrides via command line.
*   **Extensible:** Includes placeholders for potential future enhancements like automatic data format detection and visualization.

## Repository Structure

```
├── configs/               # YAML configs for different scenes (lego.yaml, mic.yaml)
├── data/                  # Sample datasets (Blender, Real-world COLMAP - use symlinks/download scripts)
├── docs/                  # Technical deep dive (ARCHITECTURE.md, BENCHMARKS.md)
├── models/                # Pretrained weights storage
├── src/                   # Source code
│   ├── __init__.py
│   ├── data_loader.py     # Data loading (Blender, LLFF) & ray generation, format inference placeholder
│   ├── hash_encoder.py    # Hash grid encoder (using tiny-cuda-nn) OR Fallback MLP
│   ├── nerf.py            # Core NeRF MLP model
│   ├── renderer.py        # Volume rendering logic (incl. hierarchical sampling)
│   ├── train.py           # Main training script using configargparse
│   └── utils.py           # Utility functions (logging, saving, PSNR, etc.)
├── Dockerfile             # Reproducible CUDA environment setup (includes tiny-cuda-nn build)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup

**1. Clone the Repository:**
```bash
git clone https://github.com/Gifted87/instant-nerf.git # Replace with your repo URL
cd instant-nerf
```

**2. Environment Setup (Option A: Docker - Recommended for Reproducibility & `tiny-cuda-nn`):**
   *   Ensure you have Docker and NVIDIA Container Toolkit installed.
   *   Build the Docker image (this installs PyTorch, dependencies, and attempts to build `tiny-cuda-nn`):
    ```bash
    docker build -t instant-nerf .
    ```
   *   Run the container (mount data/models if needed):
    ```bash
    # Mount current directory to /workspace inside container
    docker run --gpus all -it --rm -v $(pwd):/workspace instant-nerf bash
    ```

**3. Environment Setup (Option B: Manual/Conda - Requires manual `tiny-cuda-nn` install):**
   *   Create a Python environment (e.g., Conda):
    ```bash
    conda create -n instant-nerf python=3.10 -y
    conda activate instant-nerf
    ```
   *   Install PyTorch (ensure CUDA compatibility): Check [https://pytorch.org/](https://pytorch.org/) for the correct command. Example for CUDA 11.8:
    ```bash
    # Example for CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
   *   Install other Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
   *   **(Optional but Recommended for Speed)** **Install `tiny-cuda-nn`:** This requires a C++14 compatible compiler and CUDA toolkit. Follow the instructions on the [official tiny-cuda-nn repository](https://github.com/NVlabs/tiny-cuda-nn). If this step fails or is skipped, the code will automatically use a slower fallback encoder.
    ```bash
    # Example steps (adapt as needed)
    pip install ninja # Recommended build tool
    git clone --recursive https://github.com/NVlabs/tiny-cuda-nn
    cd tiny-cuda-nn/bindings/torch
    python setup.py install # Requires CUDA toolkit and C++ compiler matching PyTorch
    cd ../../../ # Go back to project root
    ```

**4. Data:**
   *   Download sample datasets (e.g., NeRF Blender Synthetic, your own COLMAP data).
   *   Place or symlink them into the `data/` directory (e.g., `data/lego`, `data/mic`).
   *   The `data_loader.py` attempts basic format inference (Blender/LLFF) but may require manual adjustment for custom data. It also estimates `scene_bound` required for normalization.

## Usage

**1. Configure:**
   *   Edit a YAML file in `configs/` (e.g., `configs/lego.yaml`).
   *   Adjust `dataset_path`, `log_dir`, `checkpoint_dir`, and other parameters as needed. Pay attention to `scene_bound` if the automatic estimation seems incorrect.

**2. Training:**
   *   Run the main training script, specifying your configuration file.
   ```bash
   python src/train.py --config configs/lego.yaml
   ```
   *   You can override parameters from the config file via command line, e.g.:
   ```bash
   python src/train.py --config configs/lego.yaml --learning_rate 0.005 --num_iterations 50000
   ```
   *   Training progress, logs, and checkpoints will be saved to the directories specified in the config (`log_dir`, `checkpoint_dir`). TensorBoard can be used to monitor training: `tensorboard --logdir logs/lego` (replace `logs/lego` with your `log_dir`).
   *   If `tiny-cuda-nn` is not installed/working, training will proceed using the fallback MLP encoder (expect significantly slower performance).

**3. Evaluation / Rendering (TODO):**
   *   *(Add script/commands for evaluating PSNR on a test set or rendering a video path using a trained checkpoint)*
   ```bash
   # Example placeholder command (needs implementation)
   # python src/evaluate.py --config configs/lego.yaml --checkpoint models/lego/final.pth --output_dir renders/lego
   ```

## Benchmarks

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for performance comparisons (primarily using the `tiny-cuda-nn` backend).

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for a technical overview of the hash grid encoding (and fallback), volume rendering, and normalization.

## Dependencies

*   PyTorch (>=1.10 recommended, tested with 2.0+)
*   `tiny-cuda-nn` (Optional, for GPU acceleration) - Requires compatible CUDA Toolkit & C++ Compiler.
*   `configargparse` (For config file + cmd line args)
*   `numpy`
*   `PyYAML`
*   `imageio` & `imageio-ffmpeg` (For image loading/saving)
*   `tqdm` (Progress bars)
*   `tensorboard` (Logging)
*   (Optional) `ninja` (For faster `tiny-cuda-nn` compilation)

See `requirements.txt` for pinned versions (excluding `torch` and `tiny-cuda-nn` which depend on your system).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
