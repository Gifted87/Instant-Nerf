import torch
import torch.nn as nn
import torch.nn.functional as F
from .hash_encoder import HashEncoder # Import the potentially fallback-enabled encoder

class NeRFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_viewdirs = config.get('use_viewdirs', True)
        # use_hash_encoder check now happens *inside* HashEncoder
        self.encoder = HashEncoder(config, input_dim=3)
        self.encoding_dim = self.encoder.output_dim

        # --- Density Network (Sigma MLP) ---
        sigma_layers = []
        in_dim = self.encoding_dim
        hidden_dim = config.get('hidden_dim_sigma', 64)
        num_layers = config.get('num_layers_sigma', 2) # NGP uses small MLPs

        layer_dims = [in_dim] + [hidden_dim] * (num_layers - 1)

        for i in range(num_layers - 1):
            sigma_layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            sigma_layers.append(nn.ReLU(inplace=True))
        
        # Separate final layer for potential feature extraction before output
        self.sigma_hidden_layer = nn.Sequential(*sigma_layers)
        self.sigma_output_layer = nn.Linear(hidden_dim, 1) # Output sigma

        # --- Color Network (RGB MLP) ---
        color_layers = []
        hidden_dim_color = config.get('hidden_dim_color', 64)
        num_layers_color = config.get('num_layers_color', 3) # NGP uses small MLPs

        # Input to color MLP comes from sigma hidden features
        color_input_dim = hidden_dim 

        if self.use_viewdirs:
             # Simple embedding or just raw directions for view dependence
             # Raw directions often work okay with TCNN MLPs
             self.dir_encoding_dim = 3 
             color_input_dim += self.dir_encoding_dim
        else:
            self.dir_encoding_dim = 0

        color_layer_dims = [color_input_dim] + [hidden_dim_color] * (num_layers_color - 1) + [3] # Output RGB

        for i in range(num_layers_color):
            color_layers.append(nn.Linear(color_layer_dims[i], color_layer_dims[i+1]))
            if i < num_layers_color - 1: # Apply ReLU to all but the last layer
                color_layers.append(nn.ReLU(inplace=True))
            else: # Apply Sigmoid to the last layer
                color_layers.append(nn.Sigmoid()) 
                
        self.color_net = nn.Sequential(*color_layers)

        # Initialize weights (optional but can help)
        self.apply(self._init_weights)
        print(f"Initialized NeRFModel (View Dirs: {self.use_viewdirs}, Encoder Output Dim: {self.encoding_dim})")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # He initialization for ReLU layers can be good
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0) # Zero bias initialization

    def forward(self, pts_norm, dirs=None):
        """
        Args:
            pts_norm (Tensor): Normalized points in 3D space (..., 3). 
                               MUST BE NORMALIZED to approx [0, 1]^3.
            dirs (Tensor, optional): Viewing directions (..., 3). 
                                     Required if use_viewdirs is True.

        Returns:
            sigma (Tensor): Density values (...)
            rgb (Tensor): Color values (..., 3)
        """
        original_shape = pts_norm.shape[:-1] 
        pts_flat = pts_norm.reshape(-1, pts_norm.shape[-1])

        # --- Encoding ---
        encoded_pts = self.encoder(pts_flat)

        # --- Sigma Prediction ---
        # Pass encoded points through density MLP
        sigma_features = self.sigma_hidden_layer(encoded_pts)
        # Get sigma output from the final layer
        sigma = self.sigma_output_layer(sigma_features)
        # Apply activation - ReLU is common for density to ensure non-negativity
        sigma = F.relu(sigma, inplace=True) 

        # --- Color Prediction ---
        color_input = sigma_features # Use features from before the final sigma layer

        if self.use_viewdirs:
            if dirs is None:
                raise ValueError("Viewing directions 'dirs' must be provided when use_viewdirs is True.")
            dirs_flat = dirs.reshape(-1, dirs.shape[-1])
            # Normalize directions (important for view dependence consistency)
            dirs_flat_norm = F.normalize(dirs_flat, p=2, dim=-1)
            # Concatenate direction encoding
            color_input = torch.cat([color_input, dirs_flat_norm], dim=-1)

        rgb = self.color_net(color_input)

        # Reshape outputs to match input spatial dimensions
        sigma = sigma.reshape(*original_shape) # Remove the last dim (was 1)
        rgb = rgb.reshape(*original_shape, 3)

        return sigma, rgb

# Example Usage
if __name__ == '__main__':
    print("-" * 80)
    print("Testing NeRFModel...")
    dummy_config = {
        'use_viewdirs': True,
        'use_hash_encoder': True, # Let HashEncoder decide if tcnn is available
        'hidden_dim_sigma': 64,
        'num_layers_sigma': 2,
        'hidden_dim_color': 64,
        'num_layers_color': 3,
        # Hash encoder params needed by HashEncoder
        'hash_num_levels': 16,
        'hash_features_per_level': 2,
        'hash_log2_size': 19,
        'hash_base_resolution': 16,
        'hash_max_resolution': 2048,
        'fallback_encoder_dim': 32, # For fallback case
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeRFModel(dummy_config).to(device)
    # print(model) # Print model structure

    # Test forward pass with dummy data
    N_rays = 5
    N_samples = 10
    # Dummy points - IMPORTANT: These should be in [0, 1]^3 range for hash grid
    test_pts_norm = torch.rand(N_rays, N_samples, 3, device=device) 
    test_dirs = F.normalize(torch.randn(N_rays, N_samples, 3, device=device), dim=-1)

    sigma, rgb = model(test_pts_norm, test_dirs)

    print(f"  Input normalized points shape: {test_pts_norm.shape}")
    print(f"  Input directions shape: {test_dirs.shape}")
    print(f"  Output sigma shape: {sigma.shape}")
    print(f"  Output rgb shape: {rgb.shape}")

    assert sigma.shape == (N_rays, N_samples)
    assert rgb.shape == (N_rays, N_samples, 3)
    print("  NeRFModel test successful.")
    print("-" * 80)