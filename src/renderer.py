import torch
import torch.nn.functional as F

# Utility function for sampling points along rays using inverse transform sampling
# Based on original NeRF implementation
def sample_pdf(bins, weights, N_samples, det=False):
    """
    Sample points from a distribution defined by weights along bins using inverse CDF.
    Args:
        bins (Tensor): Bin edges [N_rays, N_bins].
        weights (Tensor): Weights for each bin [N_rays, N_bins]. Should sum to 1 per ray ideally.
        N_samples (int): Number of samples to draw per ray.
        det (bool): Deterministic sampling (linspace) vs random sampling.
    Returns:
        samples (Tensor): Sampled z-values [N_rays, N_samples].
    """
    device = weights.device
    # Add small epsilon to weights to prevent NaNs when weights sum to 0
    weights = weights + 1e-5  # [N_rays, N_bins]
    pdf = weights / torch.sum(weights, -1, keepdim=True) # Normalize to get PDF [N_rays, N_bins]
    cdf = torch.cumsum(pdf, -1) # Cumulative distribution function [N_rays, N_bins]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # Prepend 0 for interpolation [N_rays, N_bins+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # [N_rays, N_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device) # [N_rays, N_samples]

    # Invert CDF
    u = u.contiguous()
    # Find indices where the uniform samples fall in the CDF
    inds = torch.searchsorted(cdf, u, right=True) # [N_rays, N_samples]
    
    # Clamp indices to valid range
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1)*torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # [N_rays, N_samples, 2] (indices bounding the sample)

    # Gather CDF values and bin edges corresponding to the bounding indices
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g) # [N_rays, N_samples, 2]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g) # [N_rays, N_samples, 2]

    # Linear interpolation between bin edges based on CDF values
    denom = (cdf_g[...,1] - cdf_g[...,0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom) # Avoid division by zero
    t = (u - cdf_g[...,0]) / denom # Interpolation weight [N_rays, N_samples]
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0]) # Interpolated z-values

    return samples # [N_rays, N_samples]


# Core volume rendering function
def volume_render(
    rays_o,
    rays_d,
    model,
    near,
    far,
    config,
    scene_bound=1.5, # Default scene bound, should be provided by dataset
    perturb=True,
    white_background=False,
    render_chunk_size=8192, # Process rays in chunks to manage memory
    device='cuda'
):
    """
    Performs volume rendering for a batch of rays.

    Args:
        rays_o (Tensor): Ray origins [N_rays, 3].
        rays_d (Tensor): Ray directions [N_rays, 3].
        model (nn.Module): The NeRFModel instance.
        near (float): Near plane distance.
        far (float): Far plane distance.
        config (dict): Configuration dictionary (used for sampling counts, etc.).
        scene_bound (float): The approximate radius of the scene bounding sphere centered at origin.
                             Used for normalizing points for the hash encoder.
        perturb (bool): Whether to perturb sample points along rays during training.
        white_background (bool): If True, composite against a white background.
        render_chunk_size (int): Max number of rays to process in parallel.
        device (str): Device to run on.

    Returns:
        dict: Dictionary containing rendered 'rgb_map', 'disp_map', 'acc_map', etc.
              Includes 'coarse' and 'fine' dicts if hierarchical sampling is used.
    """
    N_rays_total = rays_o.shape[0]
    use_hierarchical = config.get('use_hierarchical', True)
    N_samples_coarse = config.get('num_samples_coarse', 64)
    N_samples_fine = config.get('num_samples_fine', 128)
    use_amp = config.get('use_amp', True)

    # Chunk processing
    all_results = {'coarse': [], 'fine': []} if use_hierarchical else {'coarse': []}
    
    for i in range(0, N_rays_total, render_chunk_size):
        # Get chunk
        rays_o_chunk = rays_o[i:i+render_chunk_size]
        rays_d_chunk = rays_d[i:i+render_chunk_size]
        N_rays_chunk = rays_o_chunk.shape[0]

        # --- 1. Coarse Sampling ---
        t_vals = torch.linspace(0., 1., steps=N_samples_coarse, device=device)
        z_vals = near * (1.-t_vals) + far * (t_vals) # Sample linearly in depth
        z_vals = z_vals.expand([N_rays_chunk, N_samples_coarse])

        if perturb:
            # Stratified sampling: get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # Draw uniform samples from within intervals
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand

        # Calculate 3D points along the rays: pts = o + t*d
        pts_coarse = rays_o_chunk[...,None,:] + rays_d_chunk[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        dirs_coarse = rays_d_chunk[...,None,:].expand_as(pts_coarse) # [N_rays, N_samples, 3]

        # --- Normalize points for Hash Grid ---
        # Map points from world space [-bound, bound] to encoder space [0, 1]
        # This assumes scene is roughly centered at origin. Needs adjustment if not.
        pts_coarse_norm = (pts_coarse / scene_bound) * 0.5 + 0.5
        # Clamp ensures points outside the intended bounds are mapped to the boundary
        pts_coarse_norm = pts_coarse_norm.clamp(0.0, 1.0) 

        # Query NeRF model (coarse)
        with torch.cuda.amp.autocast(enabled=use_amp):
            sigma_coarse, rgb_coarse = model(pts_coarse_norm, dirs_coarse)

        # Render coarse results using volume rendering equation
        results_coarse_chunk = raw2outputs(sigma_coarse, rgb_coarse, z_vals, rays_d_chunk, white_background)
        all_results['coarse'].append(results_coarse_chunk)

        if not use_hierarchical:
            continue # Skip fine sampling if not enabled

        # --- 2. Fine Sampling (Hierarchical) ---
        with torch.no_grad(): # Don't need gradients for sampling process itself
             # Use weights from coarse pass to guide fine sampling
             weights_coarse = results_coarse_chunk['weights'][...,1:-1].detach() + 1e-5 # [N_rays, N_samples-2]
             
             # Calculate bin edges between coarse samples
             z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # [N_rays, N_samples-1]
             
             # Sample N_fine points using inverse CDF based on coarse weights
             z_samples_fine = sample_pdf(z_vals_mid, weights_coarse, N_samples_fine, det=(not perturb))
             z_samples_fine = z_samples_fine.detach() # [N_rays, N_samples_fine]

        # Combine coarse and fine samples and sort them
        z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples_fine], -1), -1) # [N_rays, N_coarse + N_fine]

        # Calculate 3D points for combined samples
        pts_fine = rays_o_chunk[...,None,:] + rays_d_chunk[...,None,:] * z_vals_combined[...,:,None] # [N_rays, N_total_samples, 3]
        dirs_fine = rays_d_chunk[...,None,:].expand_as(pts_fine)

        # --- Normalize points for Hash Grid (Fine) ---
        pts_fine_norm = (pts_fine / scene_bound) * 0.5 + 0.5
        pts_fine_norm = pts_fine_norm.clamp(0.0, 1.0)

        # Query NeRF model (fine) with combined samples
        with torch.cuda.amp.autocast(enabled=use_amp):
            sigma_fine, rgb_fine = model(pts_fine_norm, dirs_fine)

        # Render fine results using volume rendering equation
        results_fine_chunk = raw2outputs(sigma_fine, rgb_fine, z_vals_combined, rays_d_chunk, white_background)
        all_results['fine'].append(results_fine_chunk)

    # Concatenate results from all chunks
    final_results = {}
    for level in all_results: # 'coarse', 'fine'
        if not all_results[level]: continue # Skip if no results for this level
        level_results = {}
        # Keys are 'rgb_map', 'disp_map', etc.
        for key in all_results[level][0].keys(): 
            level_results[key] = torch.cat([r[key] for r in all_results[level]], dim=0)
        final_results[level] = level_results

    return final_results


def raw2outputs(sigma, rgb, z_vals, rays_d, white_background=False):
    """Transforms model's predictions (sigma, rgb) to rendered values.
    Args:
        sigma (Tensor): Density predictions [N_rays, N_samples].
        rgb (Tensor): Color predictions [N_rays, N_samples, 3].
        z_vals (Tensor): Sample distances along ray [N_rays, N_samples].
        rays_d (Tensor): Ray directions [N_rays, 3].
        white_background (bool): If True, assume a white background.
    Returns:
        dict containing:
            rgb_map (Tensor): Estimated RGB color [N_rays, 3].
            disp_map (Tensor): Disparity map [N_rays].
            acc_map (Tensor): Accumulated opacity [N_rays].
            weights (Tensor): Weights assigned to each sample [N_rays, N_samples].
            depth_map (Tensor): Estimated depth [N_rays].
    """
    device = sigma.device
    # Calculate distances between adjacent samples
    dists = z_vals[...,1:] - z_vals[...,:-1] # [N_rays, N_samples-1]
    # Assume distance for the last sample is infinite (or very large)
    # Use tensor notation for large distance to stay on device and dtype
    large_dist = torch.full_like(dists[...,:1], 1e10)
    dists = torch.cat([dists, large_dist], -1)  # [N_rays, N_samples]

    # Multiply distances by the norm of ray direction (accounts for non-unit directions)
    # Although directions are usually normalized beforehand
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    # Calculate alpha values (opacity) from sigma and distances
    # alpha = 1 - exp(-sigma * delta)
    alpha = 1. - torch.exp(-sigma * dists)  # [N_rays, N_samples]

    # Calculate transmittance (T) - probability of light reaching the sample
    # T_i = product_{j=1}^{i-1} (1 - alpha_j)
    # Use exclusive cumprod for efficiency: [1, 1-a1, (1-a1)(1-a2), ...]
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples]

    # Calculate weights for each sample: weight_i = T_i * alpha_i
    weights = alpha * transmittance  # [N_rays, N_samples]

    # Composite RGB values using weights: C = sum(w_i * c_i)
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    # Calculate depth map: E[t] = sum(w_i * t_i)
    depth_map = torch.sum(weights * z_vals, -1)
    
    # Calculate disparity map: 1 / E[t] (use acc_map for stability)
    # Add epsilon to avoid division by zero if acc_map is 0
    acc_map = torch.sum(weights, -1) # Total accumulated opacity
    disp_map = acc_map / (depth_map + 1e-10) # Disparity = acc / depth
    # Or: disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (acc_map + 1e-10))


    if white_background:
        # Composite RGB map against white background using accumulation map
        # Final Color = Rendered Color + (1 - Accumulated Opacity) * Background Color (1.0)
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'weights': weights, 'depth_map': depth_map}


# Helper to get rays for a whole image (used in evaluation/rendering)
def get_rays(H, W, K, c2w, device='cuda'):
    """Get ray origins, directions in world coordinates for all pixels of an image."""
    # Create grid of pixel coordinates
    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), 
                          torch.linspace(0, H-1, H, device=device), 
                          indexing='xy') # Note: H, W order might need swapping depending on convention
    
    # Convert pixel coordinates to camera coordinates using intrinsics K
    # dirs = [ (i - cx)/fx, -(j - cy)/fy, -1 ]  (Negative Z points forward)
    dirs = torch.stack([(i-K[0][2])/K[0][0], 
                        -(j-K[1][2])/K[1][1], # Y is often flipped in camera vs image coords
                        -torch.ones_like(i)], -1) # [H, W, 3]
    
    # Rotate ray directions from camera frame to the world frame using rotation part of c2w
    # rays_d = R @ dirs (where R = c2w[:3,:3])
    # Efficient batch dot product: sum(dirs[..., None, :] * R, -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # [H, W, 3]
    
    # Translate camera frame's origin (position) to the world frame. It is the origin of all rays.
    # Origin is the translation part of c2w
    rays_o = c2w[:3,-1].expand(rays_d.shape) # [H, W, 3]
    
    return rays_o, rays_d # Both [H, W, 3]