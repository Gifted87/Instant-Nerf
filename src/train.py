import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import configargparse # For handling config files and cmd args
import os
import time
from tqdm import tqdm

# Import local modules
from .data_loader import get_data_loader 
from .nerf import NeRFModel
from .renderer import volume_render
from .utils import (
    AverageMeter, 
    save_checkpoint, 
    load_checkpoint, 
    render_and_save_image, 
    mse_to_psnr
)

def main(args):
    # --- Load Config ---
    try:
        with open(args.config, 'r') as f:
            # Use configargparse's method to load yaml and merge args
            config_parser = configargparse.YAMLConfigFileParser()
            config_dict = config_parser.parse(f) 
            # Merge command line args onto config file values
            # Configargparse handles this automatically if args are defined correctly below
            config = args # Use the parsed args object which now contains merged values
            
        print("--- Configuration ---")
        # configargparse prints the args/config source, which is helpful
        # print(config) # Print the full config if needed
        print(f"  Log Dir: {config.log_dir}")
        print(f"  Checkpoint Dir: {config.checkpoint_dir}")
        print(f"  Dataset Path: {config.dataset_path}")
        print(f"  Device: {config.device}")
        print("---------------------")
        
    except Exception as e:
        print(f"ERROR loading configuration: {e}")
        return

    # --- Setup ---
    device = torch.device(config.device)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available() and 'cuda' in config.device:
        torch.cuda.manual_seed_all(config.seed)
        # torch.backends.cudnn.benchmark = True # Can speed up if input sizes don't vary much

    log_dir = config.log_dir
    checkpoint_dir = config.checkpoint_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Save effective config to log dir for reproducibility
    config_save_path = os.path.join(log_dir, 'effective_config.yaml')
    with open(config_save_path, 'w') as f:
         # Convert Namespace to dict for saving
         config_dict_to_save = vars(config)
         # We don't need to save the config file path itself within the file
         config_dict_to_save.pop('config', None) 
         yaml.dump(config_dict_to_save, f, indent=2)
         print(f"Effective config saved to {config_save_path}")


    # --- Data Loading ---
    # Training data loader samples rays directly
    train_loader, train_dataset = get_data_loader(config, split='train', device=device)
    # Validation data loader loads full images
    val_loader, val_dataset = get_data_loader(config, split='val', device=device)
    print(f"Train dataset: {len(train_dataset)} rays.")
    print(f"Val dataset: {len(val_dataset)} images.")

    # --- Model ---
    # Pass the whole config namespace to the model
    model = NeRFModel(vars(config)).to(device) 
    # print(model) # Optional: Print model structure

    # --- Optimizer and Scheduler ---
    lr = config.learning_rate 
    # NGP often uses Adam with specific betas/eps
    optimizer = optim.Adam(model.parameters(), lr=lr, 
                           betas=config.adam_betas, 
                           eps=config.adam_eps) 
    
    # NGP typically uses exponential decay, adjust gamma and step_size
    # Step size is often large (e.g., 100k-500k), gamma small (e.g., 0.1-0.5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay_gamma**(1/config.lr_decay_steps))
    # Or use StepLR:
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_steps, gamma=config.lr_decay_gamma)

    # --- Automatic Mixed Precision (AMP) ---
    use_amp = config.use_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision: {use_amp}")

    # --- Checkpoint Loading ---
    start_iter = 0
    if config.resume:
        checkpoint_path = os.path.join(checkpoint_dir, config.resume) # Try specified filename
        if not os.path.exists(checkpoint_path):
             checkpoint_path = os.path.join(checkpoint_dir, "latest.pth") # Or try latest
             
        if os.path.exists(checkpoint_path):
            start_iter = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler)
            print(f"Resumed training from iteration {start_iter}")
        else:
            print(f"No checkpoint found at {config.resume} or latest.pth. Starting from scratch.")


    # --- Training Loop ---
    num_iterations = config.num_iterations 
    print(f"Starting training for {num_iterations} iterations...")

    global_step = start_iter
    model.train()
    train_loss_meter = AverageMeter() 
    train_psnr_meter = AverageMeter()
    
    # NGP often iterates directly over steps rather than epochs
    # Pre-fetch data using iterator for potentially faster loading
    data_iter = iter(train_loader) 
    
    pbar = tqdm(total=num_iterations, initial=global_step, desc="Training")
    start_time = time.time()

    while global_step < num_iterations:
        
        # --- Data Fetching ---
        try:
            batch = next(data_iter)
        except StopIteration:
            # Epoch finished (less relevant for iteration-based training)
            # Simply restart the iterator
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Move batch to device
        rays_o = batch['rays_o'].to(device, non_blocking=True)
        rays_d = batch['rays_d'].to(device, non_blocking=True)
        target_rgb = batch['target_rgb'].to(device, non_blocking=True)

        # --- Forward Pass & Loss Calculation ---
        optimizer.zero_grad(set_to_none=True) # More memory efficient

        with torch.cuda.amp.autocast(enabled=use_amp):
            render_results = volume_render(
                rays_o, rays_d, model,
                near=train_dataset.near, 
                far=train_dataset.far,
                scene_bound=train_dataset.scene_bound, # Pass scene bound from dataset
                config=vars(config), # Pass full config
                perturb=True, # Perturb sampling during training
                white_background=config.white_background,
                render_chunk_size=config.render_chunk_size, # Use same chunk size for train
                device=device
            )
            
            # Calculate loss (typically MSE on RGB)
            # Use fine map results if available (hierarchical sampling)
            pred_rgb = render_results['fine']['rgb_map'] if 'fine' in render_results else render_results['coarse']['rgb_map']
            
            # Simple MSE loss
            loss = F.mse_loss(pred_rgb, target_rgb)

            # --- Optional: Add coarse loss if training hierarchically ---
            # Often not needed for NGP convergence but sometimes used
            if config.use_hierarchical and config.coarse_loss_weight > 0 and 'coarse' in render_results:
                 pred_rgb_coarse = render_results['coarse']['rgb_map']
                 loss_coarse = F.mse_loss(pred_rgb_coarse, target_rgb)
                 loss = loss + config.coarse_loss_weight * loss_coarse 

        # --- Backpropagation with AMP ---
        scaler.scale(loss).backward()
        # Optional: Gradient clipping (can help stability sometimes)
        if config.grad_clip_norm > 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # Step scheduler each iteration

        # --- Logging ---
        loss_item = loss.item()
        train_loss_meter.update(loss_item)
        writer.add_scalar('Loss/train', loss_item, global_step)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)
        
        # Calculate and log PSNR (using fine map if available)
        with torch.no_grad():
            psnr = mse_to_psnr(loss if config.coarse_loss_weight == 0 else F.mse_loss(pred_rgb, target_rgb)) # Use fine loss for PSNR
            train_psnr_meter.update(psnr.item())
            writer.add_scalar('Metrics/train_psnr', psnr.item(), global_step)

        # Update progress bar
        pbar.set_postfix(loss=f"{loss_item:.4f}", psnr=f"{psnr.item():.2f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        pbar.update(1)
        global_step += 1

        # --- Validation & Checkpointing ---
        if global_step % config.validate_every_iters == 0:
            val_loss, val_psnr = validate(model, val_loader, val_dataset, config, device)
            writer.add_scalar('Loss/val', val_loss, global_step)
            writer.add_scalar('Metrics/val_psnr', val_psnr, global_step)
            tqdm.write(f"\nIter {global_step}/{num_iterations} Validation: Loss={val_loss:.4f}, PSNR={val_psnr:.2f}")
            # Log average training metrics for the interval
            writer.add_scalar('Loss/train_avg', train_loss_meter.avg, global_step)
            writer.add_scalar('Metrics/train_psnr_avg', train_psnr_meter.avg, global_step)
            train_loss_meter.reset()
            train_psnr_meter.reset()
            # Save latest checkpoint after validation
            save_path = os.path.join(checkpoint_dir, "latest.pth")
            save_checkpoint(save_path, global_step, model, optimizer, scheduler, scaler)
            # Switch back to training mode
            model.train() 

        if config.save_every_iters > 0 and global_step % config.save_every_iters == 0:
             save_path = os.path.join(checkpoint_dir, f"iter_{global_step}.pth")
             save_checkpoint(save_path, global_step, model, optimizer, scheduler, scaler)

        # --- Rendering Test Views (Optional) ---
        if config.render_every_iters > 0 and global_step % config.render_every_iters == 0:
            render_test_view(model, val_dataset, config, device, global_step, log_dir)
            model.train() # Switch back to training mode


    pbar.close()
    # Final Save
    save_path = os.path.join(checkpoint_dir, "final.pth")
    save_checkpoint(save_path, global_step, model, optimizer, scheduler, scaler)
    writer.close()
    print(f"Training finished in {(time.time() - start_time)/60:.2f} minutes.")
    print(f"Final checkpoint saved to {save_path}")
    print(f"Logs saved to {log_dir}")


def validate(model, val_loader, val_dataset, config, device):
    """ Run validation loop """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_psnr = 0.0
    
    with torch.no_grad():
        pbar_val = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=False)
        for i, batch in pbar_val:
            # Data loader gives rays for one image at a time (batch_size=1 for val)
            rays_o = batch['rays_o'].squeeze(0).to(device) # (H*W, 3)
            rays_d = batch['rays_d'].squeeze(0).to(device) # (H*W, 3)
            target_rgb = batch['target_rgb'].squeeze(0).to(device) # (H*W, C)
            H = batch['height'].item()
            W = batch['width'].item()

            # Render the image using volume_render (handles chunking internally)
            render_results = volume_render(
                rays_o, rays_d, model,
                near=val_dataset.near, 
                far=val_dataset.far,
                scene_bound=val_dataset.scene_bound,
                config=vars(config),
                perturb=False, # No perturbation during validation
                white_background=config.white_background,
                render_chunk_size=config.render_chunk_size, # Use potentially larger chunk for val
                device=device
            )

            # Use fine map if available
            rgb_preds = render_results['fine']['rgb_map'] if 'fine' in render_results else render_results['coarse']['rgb_map']
            
            loss = F.mse_loss(rgb_preds, target_rgb)
            psnr = mse_to_psnr(loss)

            total_loss += loss.item()
            total_psnr += psnr.item()
            
            pbar_val.set_postfix(img_loss=f"{loss.item():.4f}", img_psnr=f"{psnr.item():.2f}")


    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    return avg_loss, avg_psnr


def render_test_view(model, val_dataset, config, device, step, log_dir):
    """ Renders a fixed validation view and saves it """
    print(f"Rendering test view at step {step}...")
    model.eval() # Set model to evaluation mode
    # Render the first validation image (index 0)
    render_idx = 0 
    if render_idx >= len(val_dataset):
         print("Warning: Validation set has no images to render.")
         return
         
    with torch.no_grad():
        batch = val_dataset[render_idx] # Get data for one image using __getitem__

        rays_o = batch['rays_o'].to(device) # (H*W, 3)
        rays_d = batch['rays_d'].to(device) # (H*W, 3)
        H = batch['height']
        W = batch['width']
        
        # Render the image
        render_results = volume_render(
            rays_o, rays_d, model,
            near=val_dataset.near, 
            far=val_dataset.far,
            scene_bound=val_dataset.scene_bound,
            config=vars(config),
            perturb=False,
            white_background=config.white_background,
            render_chunk_size=config.render_chunk_size,
            device=device
        )
        
        rgb_preds = render_results['fine']['rgb_map'] if 'fine' in render_results else render_results['coarse']['rgb_map']
        rgb_preds = rgb_preds.reshape(H, W, 3) # Reshape back to image format

        # Save rendered image using utility function
        render_save_path = os.path.join(log_dir, f"render_step_{step:06d}.png")
        render_and_save_image(rgb_preds, render_save_path)
        
        # Save target image only once for comparison (optional)
        target_save_path = os.path.join(log_dir, f"target_{batch['index']:03d}.png")
        if not os.path.exists(target_save_path) and 'target_rgb' in batch:
            target_rgb = batch['target_rgb'].to(device).reshape(H, W, -1) # (H, W, C)
            render_and_save_image(target_rgb, target_save_path)
            
    print(f"Test view saved to {render_save_path}")


if __name__ == "__main__":
    # Use configargparse for combined yaml/command-line configuration
    parser = configargparse.ArgumentParser(
        description="Train an Instant-NGP accelerated NeRF model.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Essential Arguments ---
    parser.add_argument('--config', is_config_file=True, required=True,
                        help='Path to the configuration file (e.g., configs/lego.yaml)')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the dataset directory')
    parser.add_argument('--log_dir', type=str, required=True, 
                        help='Directory to save logs and TensorBoard summaries')
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                        help='Directory to save model checkpoints')

    # --- Training Parameters ---
    parser.add_argument('--num_iterations', type=int, default=25000, 
                        help='Total number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                        help='Initial learning rate')
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.99], 
                        help='Adam optimizer beta parameters')
    parser.add_argument('--adam_eps', type=float, default=1e-15, 
                        help='Adam optimizer epsilon')
    parser.add_argument('--lr_decay_steps', type=int, default=500000, 
                        help='Number of steps for one full decay cycle (ExponentialLR)')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1, 
                        help='Learning rate decay factor (over decay_steps)')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0, 
                        help='Gradient clipping norm value (0.0 for no clipping)')
    parser.add_argument('--ray_batch_size', type=int, default=4096, 
                        help='Number of rays to sample per training batch')
    parser.add_argument('--coarse_loss_weight', type=float, default=0.0, 
                        help='Weight for the coarse loss term (0.0 to disable)')

    # --- Model Parameters (Passed via config dict to NeRFModel) ---
    parser.add_argument('--use_viewdirs', action='store_true', default=True, 
                        help='Use viewing directions in the model')
    parser.add_argument('--hidden_dim_sigma', type=int, default=64, help='Hidden dimension for density MLP')
    parser.add_argument('--num_layers_sigma', type=int, default=2, help='Number of layers for density MLP')
    parser.add_argument('--hidden_dim_color', type=int, default=64, help='Hidden dimension for color MLP')
    parser.add_argument('--num_layers_color', type=int, default=3, help='Number of layers for color MLP')

    # --- Hash Encoder Parameters ---
    parser.add_argument('--use_hash_encoder', action='store_true', default=True, help='Use TCNN HashGrid encoder if available')
    parser.add_argument('--hash_log2_size', type=int, default=19, help='Log2 of hash table size (T)')
    parser.add_argument('--hash_num_levels', type=int, default=16, help='Number of levels in hash grid (L)')
    parser.add_argument('--hash_features_per_level', type=int, default=2, help='Features per level (F)')
    parser.add_argument('--hash_base_resolution', type=int, default=16, help='Base resolution (N_min)')
    parser.add_argument('--hash_max_resolution', type=int, default=2048, help='Max resolution (N_max)')
    parser.add_argument('--fallback_encoder_dim', type=int, default=32, help='Output dimension for fallback MLP encoder')


    # --- Renderer Parameters ---
    parser.add_argument('--near_bound', type=float, help='Override near bound (otherwise inferred from data)')
    parser.add_argument('--far_bound', type=float, help='Override far bound (otherwise inferred from data)')
    parser.add_argument('--scene_bound', type=float, help='Override scene bound radius (otherwise estimated)')
    parser.add_argument('--num_samples_coarse', type=int, default=64, help='Number of coarse samples per ray')
    parser.add_argument('--num_samples_fine', type=int, default=128, help='Number of fine samples per ray')
    parser.add_argument('--use_hierarchical', action='store_true', default=True, help='Use hierarchical sampling')
    parser.add_argument('--white_background', action='store_true', default=False, 
                        help='Assume white background during rendering (for synthetic data)')
    parser.add_argument('--render_chunk_size', type=int, default=8192, 
                        help='Chunk size for rendering rays (adjust based on GPU memory)')

    # --- Dataset Parameters ---
    parser.add_argument('--dataset_type', type=str, default=None, 
                        help='Type of dataset (blender, llff, etc.). If None, will try to infer.')
    parser.add_argument('--half_res', action='store_true', default=False, 
                        help='Load data at half resolution (for Blender/LLFF)')
    parser.add_argument('--llff_factor', type=int, default=8, help='Downsampling factor for LLFF data')
    parser.add_argument('--testskip', type=int, default=8, 
                         help='Skip factor for loading Blender test/val data (1 = load all)')
    parser.add_argument('--near_factor', type=float, default=0.9, help='Factor to multiply inferred near bound by (LLFF)')
    parser.add_argument('--far_factor', type=float, default=1.1, help='Factor to multiply inferred far bound by (LLFF)')
    parser.add_argument('--visualize_data', action='store_true', default=False, 
                        help='Run placeholder data visualization before training')


    # --- Hardware & Logging ---
    parser.add_argument('--device', type=str, default='cuda', help='Compute device (e.g., cuda, cuda:0, cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use Automatic Mixed Precision')
    parser.add_argument('--validate_every_iters', type=int, default=5000, 
                        help='Run validation every N iterations')
    parser.add_argument('--save_every_iters', type=int, default=5000, 
                        help='Save a checkpoint every N iterations (0 to disable intermediate saves)')
    parser.add_argument('--render_every_iters', type=int, default=5000, 
                        help='Render a test view every N iterations (0 to disable)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint filename (in checkpoint_dir) to resume from (e.g., latest.pth or iter_10000.pth)')
    parser.add_argument('--train_num_workers', type=int, default=4, help='Number of worker processes for training DataLoader')
    parser.add_argument('--val_num_workers', type=int, default=0, help='Number of worker processes for validation DataLoader')


    args = parser.parse_args()
    
    # --- Basic Validation ---
    if not torch.cuda.is_available() and 'cuda' in args.device:
        print("Warning: CUDA not available, switching device to CPU.")
        args.device = 'cpu'
        args.use_amp = False # AMP only works on CUDA

    main(args)