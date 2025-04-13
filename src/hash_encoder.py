import torch
import torch.nn as nn

# --- Novelty: Graceful Fallback ---
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
    print("tinycudann library found.")
except ImportError:
    TCNN_AVAILABLE = False
    print("=" * 80)
    print("Warning: tinycudann library not found or failed to import.")
    print("         HashEncoder will use a fallback MLP implementation.")
    print("         Install tinycudann for GPU acceleration and NGP performance:")
    print("         https://github.com/NVlabs/tiny-cuda-nn")
    print("=" * 80)
    tcnn = None # Define tcnn as None if import fails

class HashEncoder(nn.Module):
    def __init__(self, config, input_dim=3):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.use_tcnn = TCNN_AVAILABLE and config.get('use_hash_encoder', True)

        if self.use_tcnn:
            print("Initializing tinycudann HashGrid Encoder...")
            # Configuration for tiny-cuda-nn HashGrid encoder
            encoder_config = {
                "otype": "HashGrid", # Grid type
                "n_levels": config.get('hash_num_levels', 16), # L
                "n_features_per_level": config.get('hash_features_per_level', 2), # F
                "log2_hashmap_size": config.get('hash_log2_size', 19), # T
                "base_resolution": config.get('hash_base_resolution', 16), # N_min
                "per_level_scale": self._calculate_per_level_scale(), # b
                "interpolation": "Linear" # Interpolation method
            }
            # The output dimension is L * F
            self.output_dim = encoder_config["n_levels"] * encoder_config["n_features_per_level"]

            try:
                self.encoder = tcnn.Encoding(
                    n_input_dims=self.input_dim,
                    encoding_config=encoder_config,
                    # Use float32 unless AMP requires float16 inputs specifically
                    dtype=torch.float32 
                )
                print(f"  Initialized tinycudann HashGrid.")
                print(f"  Output dimension: {self.output_dim}")
                # print(f"  Config: {encoder_config}") # Verbose: print full config
            except Exception as e:
                 print(f"ERROR initializing tinycudann Encoding: {e}")
                 print("Falling back to MLP encoder.")
                 self.use_tcnn = False # Force fallback if tcnn init fails
                 self._init_fallback_encoder()

        else:
            print("Using fallback MLP encoder (No tinycudann or use_hash_encoder=False).")
            self._init_fallback_encoder()

    def _init_fallback_encoder(self):
        # Simple MLP as a fallback
        # Output dim should be reasonable, e.g., 32 or 64
        self.output_dim = self.config.get('fallback_encoder_dim', 32) 
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.output_dim)
        )
        print(f"  Initialized Fallback MLP Encoder. Output dim: {self.output_dim}")

    def _calculate_per_level_scale(self):
        """ Calculate the per_level_scale (b) for the hash grid """
        n_min = self.config.get('hash_base_resolution', 16)
        n_max = self.config.get('hash_max_resolution', 2048)
        L = self.config.get('hash_num_levels', 16)
        if L <= 1:
             print("Warning: hash_num_levels <= 1, setting per_level_scale to 1.0")
             return 1.0 # Avoid division by zero or log(1)
        
        # Ensure n_max > n_min
        if n_max <= n_min:
            print(f"Warning: hash_max_resolution ({n_max}) <= hash_base_resolution ({n_min}). Adjusting max slightly.")
            n_max = n_min * 1.1 # Ensure n_max > n_min to avoid NaN/Inf in log

        try:
            b = torch.exp((torch.log(torch.tensor(float(n_max))) - torch.log(torch.tensor(float(n_min)))) / (L - 1)).item()
        except ValueError as e:
            print(f"Error calculating per_level_scale with N_min={n_min}, N_max={n_max}, L={L}: {e}")
            print("Using default scale of 1.5")
            b = 1.5 # Sensible default if calculation fails
            
        # print(f"  Calculated per_level_scale (b): {b:.4f}")
        return b

    def forward(self, x):
        """
        Encode input points x.
        Args:
            x: (..., input_dim) tensor of points.
               IMPORTANT: For HashGrid, expects points roughly in [0, 1]^input_dim.
                          Normalization must happen *before* calling this encoder.
        Returns:
            (..., output_dim) encoded features.
        """
        if not x.is_contiguous():
             x = x.contiguous()
             
        # Add check for input range if using tcnn (optional, for debugging)
        # if self.use_tcnn and self.training: # Only check during training if needed
        #     if x.min() < -0.1 or x.max() > 1.1:
        #          print(f"Warning: HashEncoder input range [{x.min():.2f}, {x.max():.2f}] might be outside optimal [0, 1].")

        return self.encoder(x)

# Example usage (for testing)
if __name__ == '__main__':
    print("-" * 80)
    print("Testing HashEncoder...")
    # Dummy config matching lego.yaml
    dummy_config_tcnn = {
        'use_hash_encoder': True, # Try to use tcnn
        'hash_num_levels': 16,
        'hash_features_per_level': 2,
        'hash_log2_size': 19,
        'hash_base_resolution': 16,
        'hash_max_resolution': 2048,
    }
    dummy_config_fallback = {
         'use_hash_encoder': False, # Force fallback
         'fallback_encoder_dim': 32,
    }

    print("\nAttempting TCNN initialization:")
    encoder_tcnn = HashEncoder(dummy_config_tcnn, input_dim=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_tcnn.to(device)
    test_input = torch.rand(10, 3, device=device) # Test points in [0, 1]^3
    encoded_output_tcnn = encoder_tcnn(test_input)
    print(f"  TCNN Input shape: {test_input.shape}")
    print(f"  TCNN Output shape: {encoded_output_tcnn.shape}")
    assert encoded_output_tcnn.shape == (10, encoder_tcnn.output_dim)
    print("  TCNN test forward pass successful.")

    print("\nAttempting Fallback MLP initialization:")
    encoder_fallback = HashEncoder(dummy_config_fallback, input_dim=3)
    encoder_fallback.to(device)
    encoded_output_fallback = encoder_fallback(test_input) # Use same input
    print(f"  Fallback Input shape: {test_input.shape}")
    print(f"  Fallback Output shape: {encoded_output_fallback.shape}")
    assert encoded_output_fallback.shape == (10, encoder_fallback.output_dim)
    print("  Fallback test forward pass successful.")
    print("-" * 80)