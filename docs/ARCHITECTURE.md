
# Instant-NeRF Architecture

This document provides a high-level overview of the technical components used in this Instant-NGP accelerated NeRF implementation.

## 1. Core NeRF Concept

Neural Radiance Fields represent a scene as a continuous volumetric function $F_{\Theta}: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$, parameterized by a neural network $\Theta$. Given a 3D point $\mathbf{x} = (x, y, z)$ and a viewing direction $\mathbf{d} = (\theta, \phi)$, the function outputs an RGB color $\mathbf{c}$ and a volume density $\sigma$.

## 2. Volume Rendering

To render a pixel, we cast a ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ from the camera origin $\mathbf{o}$ through the pixel in direction $\mathbf{d}$. The color of the ray is computed by integrating the color and density along the ray from the near plane $t_n$ to the far plane $t_f$:

$$
\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt
$$

where $T(t)$ is the accumulated transmittance:

$$
T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right)
$$

In practice, this integral is approximated numerically using quadrature rules (sampling points along the ray). This implementation (`src/renderer.py`) uses stratified sampling for the coarse pass and PDF-based sampling (inverse transform sampling) for the fine pass (hierarchical sampling). Rendering is performed in chunks (`render_chunk_size`) to manage GPU memory.

## 3. Point Encoding: Hash Grid vs. Fallback MLP

A key component is encoding the 3D point coordinates $\mathbf{x}$ into a feature vector suitable for the downstream MLP.

### 3.1 Hash Grid Encoding (Instant-NGP via `tiny-cuda-nn`)

This is the preferred, high-performance backend. When `tiny-cuda-nn` is available and `use_hash_encoder=True` in the config:

*   **Multi-Resolution Grids:** Maintain multiple hash grids ($L$ levels) at different resolutions.
*   **Hash Function:** Input coordinates $\mathbf{x}$ (***critically, normalized to approximately `[0, 1]^3`***) are hashed to find entries in each grid level's hash table ($T$ entries per table).
*   **Feature Vectors:** Each hash table entry stores a small trainable feature vector ($F$ features per entry).
*   **Interpolation:** Features from the surrounding grid vertices at each level are interpolated (trilinear interpolation) based on the point's normalized position.
*   **Concatenation:** The interpolated features from all $L$ levels are concatenated (total dimension $L \times F$) and passed to the NeRF MLP.

This allows a much smaller MLP to represent high-frequency details effectively, leading to significantly faster training compared to traditional positional encoding.

### 3.2 Fallback MLP Encoder

If `tiny-cuda-nn` is not installed, fails to import/initialize, or if `use_hash_encoder=False` in the config, the `HashEncoder` class (`src/hash_encoder.py`) automatically switches to a standard PyTorch MLP encoder:

*   **Input:** Raw (or potentially normalized, depending on MLP design) 3D coordinates $\mathbf{x}$.
*   **Architecture:** A simple sequential MLP (e.g., `Linear -> ReLU -> Linear -> ReLU -> Linear`) defined in PyTorch.
*   **Output:** A feature vector (e.g., 32 dimensions) passed to the NeRF MLP.

This ensures the code remains runnable even without the specialized CUDA library, albeit at the cost of performance (significantly slower training and potentially lower final quality for the same iteration count).

## 4. Point Coordinate Normalization

**Crucially, the `tiny-cuda-nn` Hash Grid encoder expects input coordinates $\mathbf{x}$ to be within the range `[0, 1]^3` for optimal performance.**

*   The `NeRFDataset` class (`src/data_loader.py`) attempts to estimate a `scene_bound` parameter (a rough radius around the origin encompassing the scene) based on camera positions and dataset bounds (near/far planes).
*   During rendering (`src/renderer.py`), sampled world-space points `pts` are normalized using this `scene_bound` before being passed to the `model.encoder`:
    ```python
    # Simplified normalization mapping [-bound, bound] -> [0, 1]
    pts_norm = (pts / scene_bound) * 0.5 + 0.5
    pts_norm = pts_norm.clamp(0.0, 1.0) # Clamp to ensure range
    encoded_pts = model.encoder(pts_norm)
    ```
*   Accurate `scene_bound` estimation or manual configuration is important for achieving good results with the hash grid.

## 5. NeRF MLP Structure

The core NeRF network (`src/nerf.py`) consists of two small MLPs:

*   **Density Head:** Takes the encoded features (from hash grid or fallback MLP) and outputs the volume density $\sigma$ (after ReLU activation).
    ```
    encoded_features -> MLP_sigma -> sigma (>=0)
    ```
*   **Color Head:** Takes intermediate features from the density head *and* (optionally, if `use_viewdirs=True`) normalized view directions $\mathbf{d}$ to output the RGB color $\mathbf{c}$ (after Sigmoid activation).
    ```
    (density_features, normalized_direction) -> MLP_color -> RGB ([0, 1]^3)
    ```
The MLPs are typically smaller than in vanilla NeRF when using the hash grid encoding (e.g., 2 layers for density, 3 for color).

## 6. Hierarchical Sampling

To focus computation on relevant parts of the scene:

1.  **Coarse Pass:** Sample $N_c$ points along the ray (stratified sampling). Evaluate the NeRF model (coarse pass) to get initial weights $w_i = T_i \alpha_i$.
2.  **Fine Sampling:** Sample $N_f$ additional points using inverse transform sampling based on the probability distribution defined by the coarse weights.
3.  **Fine Pass:** Evaluate the NeRF model at the combined set of $N_c + N_f$ points. The final color is rendered using the results from this combined (fine) set.

## 7. Implementation Details

*   **PyTorch:** Used for model definition, automatic differentiation, and training loop.
*   **`tiny-cuda-nn`:** (Optional) Provides the GPU-accelerated hash grid implementation.
*   **`configargparse`:** Handles configuration loading from YAML files and merging with command-line arguments (`src/train.py`).
*   **Automatic Mixed Precision (AMP):** `torch.cuda.amp` is used (if enabled) to accelerate training using FP16/BF16, especially beneficial with `tiny-cuda-nn`.
*   **Chunked Rendering:** The `volume_render` function processes rays in chunks to avoid GPU Out-of-Memory errors with large numbers of rays.
*   **Data Loader:** Precomputes all rays for the training set for faster batch sampling during training (`src/data_loader.py`).