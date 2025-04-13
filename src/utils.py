import torch
import torch.nn.functional as F
import numpy as np
import os
import imageio

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(path, global_step, model, optimizer, scheduler, scaler):
    """Saves model, optimizer, scheduler, scaler state."""
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at step {global_step}")

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """Loads checkpoint and returns the global step to resume from."""
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage) # Load to CPU first
    
    global_step = checkpoint.get('global_step', 0)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
         print("Warning: Optimizer state not found or not loaded.")
         
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("Warning: Scheduler state not found or not loaded.")
        
    # Handle scaler state (might not exist in older checkpoints)
    if scaler and 'scaler_state_dict' in checkpoint:
         scaler.load_state_dict(checkpoint['scaler_state_dict'])
         print("Loaded GradScaler state.")
    elif scaler:
        print("Warning: GradScaler state not found in checkpoint. Initializing anew.")

    # Move model back to appropriate device after loading state dict
    # Assuming device is handled outside this function (e.g., in train.py)
    # model.to(device) 
        
    print(f"Resuming from global step {global_step}")
    return global_step

def mse_to_psnr(mse):
    """Converts Mean Squared Error to Peak Signal-to-Noise Ratio."""
    if mse == 0:
        return float('inf')
    return -10. * torch.log10(mse)

def render_and_save_image(tensor_image, filepath):
    """
    Renders a tensor image (H, W, C) [0, 1] float to a PNG file.
    """
    try:
        # Ensure tensor is on CPU and detached from graph
        image_numpy = tensor_image.detach().cpu().numpy()
        # Clamp values and convert to uint8
        image_numpy = np.clip(image_numpy * 255.0, 0, 255).astype(np.uint8)
        # Save image
        imageio.imwrite(filepath, image_numpy)
        # print(f"Image saved to {filepath}")
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")


def infer_dataset_format(datadir):
    """
    Placeholder for AI-Powered Data Parser.
    Currently uses simple heuristics.
    """
    if os.path.exists(os.path.join(datadir, 'transforms_train.json')):
        print("Inferred Blender dataset format.")
        return 'blender'
    elif os.path.exists(os.path.join(datadir, 'poses_bounds.npy')):
        print("Inferred LLFF dataset format.")
        return 'llff'
    # Add more heuristics for COLMAP structures etc.
    else:
        print("Warning: Could not infer dataset format. Assuming custom.")
        return 'custom' # Or raise an error

# --- Novelty Idea Placeholder ---
def visualize_data_loading(poses, H, W, K, sparse_pts=None, estimated_bounds=None, save_path="docs/data_visualization.png"):
     """ 
     Placeholder for "Data Doctor" Visualization Tool.
     Requires integrating a 3D plotting library like open3d or plotly.
     """
     print(f"Placeholder: Visualize data - {poses.shape[0]} cameras, resolution {H}x{W}.")
     print(f"Visualization would be saved to {save_path}")
     # Example using print statements:
     print(f"  Intrinsics (K):\n{K}")
     if estimated_bounds:
         print(f"  Estimated Scene Bounds (Center, Scale): {estimated_bounds}")
     # --- Add actual 3D plotting logic here ---
     # e.g., plot camera frustums using K and poses
     # e.g., plot sparse points if available
     # e.g., plot the estimated bounding box