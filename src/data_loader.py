import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import imageio.v3 as iio
import numpy as np
import glob
from .utils import infer_dataset_format, visualize_data_loading # Import novelty placeholders


# --- Placeholder Data Loading Functions ---
# These need full implementations based on dataset specifics.

def load_blender_data(basedir, half_res=False, testskip=1):
    """Loads Blender dataset (NeRF JSON format)."""
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f'transforms_{s}.json'), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(iio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        # Convert images to [0, 1] float32
        imgs = (np.array(imgs) / 255.).astype(np.float32) 
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs.shape[1:3]
    camera_angle_x = float(metas['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([torch.from_numpy(pose_spherical(angle, -30.0, 4.0)) 
                                for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
         # Resize images and adjust focal length
         H = H//2
         W = W//2
         focal = focal/2.
         # Could use PIL or OpenCV for resizing if needed, requires extra dependency
         # For simplicity, just use numpy slicing (crude downsampling)
         imgs_half_res = np.zeros((imgs.shape[0], H, W, imgs.shape[3]))
         for i, img in enumerate(imgs):
              # Simple block averaging for downsampling RGBA
              # This is basic, cv2.resize(INTER_AREA) is better
              reshaped = img.reshape(H, 2, W, 2, 4)
              imgs_half_res[i] = reshaped.mean(axis=(1, 3))
         imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal], i_split


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False):
    """Loads LLFF dataset (poses_bounds.npy format)."""
    # Implementation adapted from original NeRF code (nerf-pytorch repo often used as reference)
    # This is complex and involves pose loading, coordinate transformations, etc.
    # --- Start of Reference LLFF Loading Logic ---
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # Nx3x5 -> 3x5xN
    bds = poses_arr[:, -2:].transpose([1,0]) # Nx2 -> 2xN
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = iio.imread(img0).shape
    
    sfx = ''
    if factor is not None:
        sfx = f'_{factor}'
        imgdir = os.path.join(basedir, 'images' + sfx)
        if not os.path.exists(imgdir):
            print( imgdir, 'does not exist, returning' )
            # Need logic to downsample images if images_N folder doesn't exist
            # raise FileNotFoundError(f"Downsampled images not found: {imgdir}")
            print(f"ERROR: Downsampled images not found: {imgdir}")
            print("Please run colmap_downsample.py or similar script first.")
            return None # Or raise error

    else:
        imgdir = os.path.join(basedir, 'images')
                            
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( f'Mismatch between imgs {len(imgfiles)} and poses {poses.shape[-1]} !!!!' )
        return

    sh = iio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # Store H, W in poses
    poses[2, 4, :] = poses[2, 4, :] * 1./factor # Adjust focal length for downsampling factor
    
    imgs = [iio.imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1) # (H, W, 3, N)
    
    # Correct pose matrix format from LLFF conventions to standard NeRF
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # Adjust coordinate system
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # (N, 3, 5)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32) # (N, H, W, 3)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32) # (N, 2)
    
    # Rescale poses so that the average distance from camera origins to the center is 1.0
    scale = 1. if not recenter else 1./(np.percentile(np.abs(poses[:,:3,3]), 50) + 1e-8)
    poses[:,:3,3] *= scale 
    bds *= scale
    
    if recenter:
        poses = recenter_poses(poses)

    # Generate render poses (often spherical for evaluation)
    render_poses = generate_render_poses(poses, bds) # Needs implementation

    # --- End of Reference LLFF Loading Logic ---

    # Need to determine i_test split (e.g., every 8th image)
    i_test = np.arange(images.shape[0])[::8] # Example split

    print(f"Loaded LLFF data: {images.shape[0]} images.")
    return images, poses, bds, render_poses, i_test

# Helper for LLFF loading
def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def normalize(v):
    return v/np.linalg.norm(v)
    
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

# Helper for LLFF/Blender loading
def pose_spherical(angle, elev, radius):
    # Standard NeRF spherical pose generation
    c2w = trans_t(radius) @ rot_phi(angle/180.*np.pi) @ rot_theta(elev/180.*np.pi)
    return c2w[:3,:4] # Return 3x4 matrix

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()
    
# Helper for LLFF Render Pose Generation
def generate_render_poses(poses, bds):
     # Placeholder: Generate spherical poses based on loaded poses/bounds
     # This logic is often specific to how the scene was captured.
     # Using a generic spherical path for now.
     c2w = poses_avg(poses)
     up = normalize(poses[:, :3, 1].sum(0))
     tt = poses[:,:3,3] 
     rads = np.percentile(np.abs(tt), 90, 0)
     rads = np.array(list(rads) + [1.]) * 1.1 # Example radius calculation
     
     render_poses = []
     for angle in np.linspace(-180, 180, 40 + 1)[:-1]:
         # Example: Fixed elevation, radius based on percentile
         render_poses.append(pose_spherical(angle, -30.0, rads.mean() * 1.2 )) 
     return torch.stack(render_poses, 0)


# --- NeRF Dataset Class ---

class NeRFDataset(Dataset):
    def __init__(self, config, split='train', device='cuda'):
        super().__init__()
        self.config = config
        self.split = split
        self.device = device
        self.load_data()

    def load_data(self):
        basedir = self.config.get('dataset_path')
        # --- Novelty: Infer Format ---
        dataset_type = self.config.get('dataset_type', infer_dataset_format(basedir))
        print(f"Loading dataset type: {dataset_type} from {basedir}")
        
        half_res = self.config.get('half_res', False) # For Blender/LLFF

        if dataset_type == 'blender':
            self.images, self.poses, self.render_poses, self.hwf, self.i_split = \
                load_blender_data(basedir, half_res=half_res, testskip=self.config.get('testskip', 1))
            self.i_train, self.i_val, self.i_test = self.i_split
            self.near = self.config.get('near_bound', 2.0) # Use config or default
            self.far = self.config.get('far_bound', 6.0)
             # --- Scene Bound Estimation (Simple for Blender) ---
            self.scene_bound = self.config.get('scene_bound', 1.5) # Approx bound for synthetic scenes
            print(f"  Using scene bound: {self.scene_bound}")
            
            if self.images.shape[-1] == 4: # RGBA
                 # Pre-multiply alpha for Blender scenes if using white_bkgd during training
                if self.config.get('white_background', False):
                    self.images = self.images[..., :3] * self.images[..., -1:] + (1.0 - self.images[..., -1:])
                else:
                    self.images = self.images[..., :3] # Just take RGB

        elif dataset_type == 'llff':
            # LLFF factor often controls resolution
            factor = 2 if half_res else self.config.get('llff_factor', 8) 
            self.images, self.poses, self.bds, self.render_poses, self.i_test = \
                load_llff_data(basedir, factor=factor, recenter=True) 
            
            if self.images is None: # Handle loading failure in load_llff_data
                 raise RuntimeError(f"Failed to load LLFF data from {basedir}")

            hwf = self.poses[0, :3, -1] # Height, Width, Focal from loaded poses
            self.poses = self.poses[:, :3, :4] # Keep only 3x4 rotation/translation
            num_images = self.images.shape[0]
            
            # Use provided test indices, simple val split (e.g., same as test)
            self.i_val = self.i_test
            self.i_train = np.array([i for i in np.arange(int(num_images)) if (i not in self.i_test and i not in self.i_val)])
            
            # Near/Far bounds from loaded bds
            self.near = np.min(self.bds) * self.config.get('near_factor', 0.9) # Configurable factor
            self.far = np.max(self.bds) * self.config.get('far_factor', 1.1)
            self.hwf = [int(hwf[0]), int(hwf[1]), hwf[2]]
            
            # --- Scene Bound Estimation (Simple for LLFF) ---
            # Estimate bound based on camera positions and near/far
            cam_origins = self.poses[:, :3, 3]
            max_radius = np.max(np.linalg.norm(cam_origins, axis=1))
            # Consider far bound as well
            self.scene_bound = max(max_radius, self.far) * 1.1 # Add a margin
            self.scene_bound = self.config.get('scene_bound', self.scene_bound) # Allow override
            print(f"  Estimated scene bound: {self.scene_bound:.2f} (Near: {self.near:.2f}, Far: {self.far:.2f})")

        else:
            # Add logic for 'custom' or other types
            raise ValueError(f"Unknown or unsupported dataset type: {dataset_type}")

        # Convert numpy arrays to torch tensors
        self.images = torch.from_numpy(self.images).float()
        self.poses = torch.from_numpy(self.poses).float()
        # Ensure render_poses is a tensor
        if not isinstance(self.render_poses, torch.Tensor):
             self.render_poses = torch.from_numpy(self.render_poses).float()

        # Get H, W, Focal and compute intrinsics matrix K
        self.height, self.width, self.focal = self.hwf
        self.K = torch.tensor([
            [self.focal, 0, 0.5 * self.width],
            [0, self.focal, 0.5 * self.height],
            [0, 0, 1]
        ]).float()

        # --- Novelty: Data Doctor Visualization (Placeholder Call) ---
        if self.config.get('visualize_data', False):
             visualize_data_loading(
                 self.poses.numpy(), 
                 self.height, self.width, self.K.numpy(),
                 estimated_bounds=f"Radius ~ {self.scene_bound:.2f}"
             )

        # Select indices for the current split
        if self.split == 'train':
            self.indices = self.i_train
        elif self.split == 'val':
            self.indices = self.i_val
        elif self.split == 'test':
            self.indices = self.i_test
        elif self.split == 'render': # Special split for using render_poses
             self.indices = np.arange(self.render_poses.shape[0])
             self.poses = self.render_poses # Use render poses for this split
             self.images = None # No ground truth images for render poses
        else:
            raise ValueError(f"Unknown split: {self.split}")
            
        # --- Precompute Rays (Crucial for NGP Speed) ---
        self.generate_rays()


    def generate_rays(self):
        """ Precompute rays (origins, directions) and colors for the selected split """
        if self.split == 'render':
             print(f"Generating rays for rendering ({len(self.indices)} poses)...")
        else:
             print(f"Generating rays for {self.split} split ({len(self.indices)} images)...")
             
        self.all_rays_o = []
        self.all_rays_d = []
        self.all_rgbs = [] if self.images is not None else None

        selected_poses = self.poses[self.indices]
        selected_images = self.images[self.indices] if self.images is not None else None

        # Generate rays image by image
        for i in range(selected_poses.shape[0]):
            c2w = selected_poses[i].to(self.device) # Camera-to-world matrix
            
            # Use helper function to get rays for this pose
            rays_o, rays_d = get_rays(self.height, self.width, self.K, c2w, device=self.device) # H, W, 3
            
            # Flatten rays and store
            self.all_rays_o.append(rays_o.reshape(-1, 3)) # (H*W, 3)
            self.all_rays_d.append(rays_d.reshape(-1, 3)) # (H*W, 3)

            if selected_images is not None:
                image = selected_images[i].to(self.device) # H, W, C
                self.all_rgbs.append(image.reshape(-1, image.shape[-1])) # (H*W, C)

        # Concatenate all rays and colors
        self.all_rays_o = torch.cat(self.all_rays_o, dim=0) # (N*H*W, 3)
        self.all_rays_d = torch.cat(self.all_rays_d, dim=0) # (N*H*W, 3)
        if self.all_rgbs is not None:
            self.all_rgbs = torch.cat(self.all_rgbs, dim=0) # (N*H*W, C)
            print(f"  Generated {self.all_rays_o.shape[0]} rays with RGB targets.")
        else:
             print(f"  Generated {self.all_rays_o.shape[0]} rays for rendering.")


    def __len__(self):
        # For training, we sample rays directly from the precomputed list
        if self.split == 'train':
            return self.all_rays_o.shape[0] # Total number of rays
        else:
            # For val/test/render, length is the number of images/poses
            return len(self.indices)

    def __getitem__(self, idx):
        if self.split == 'train':
            # Return a single ray origin, direction, and its RGB color
            # The DataLoader with shuffle=True will handle random sampling of rays
            return {
                'rays_o': self.all_rays_o[idx],
                'rays_d': self.all_rays_d[idx],
                'target_rgb': self.all_rgbs[idx]
            }
        else: # val / test / render
             # Return all rays for a single image/pose index 'idx'
             # Calculate start and end indices for the rays of this image
             start_idx = idx * self.height * self.width
             end_idx = start_idx + self.height * self.width
             
             batch = {
                'rays_o': self.all_rays_o[start_idx:end_idx], # (H*W, 3)
                'rays_d': self.all_rays_d[start_idx:end_idx], # (H*W, 3)
                'height': self.height,
                'width': self.width,
                'index': self.indices[idx] # Original index of the image/pose
             }
             if self.all_rgbs is not None:
                 batch['target_rgb'] = self.all_rgbs[start_idx:end_idx] # (H*W, C)
                 
             return batch


def get_data_loader(config, split='train', batch_size=None, device='cuda'):
    """Creates and returns a DataLoader for the NeRFDataset."""
    dataset = NeRFDataset(config, split=split, device=device)
    
    # Determine batch size based on split
    if split == 'train':
        # Use ray batch size for training
        effective_batch_size = batch_size if batch_size is not None else config.get('ray_batch_size', 4096)
        shuffle = True
        num_workers = config.get('train_num_workers', 4) # More workers for training
    else:
        # Use image batch size (usually 1) for val/test/render
        effective_batch_size = 1 
        shuffle = False # Process images sequentially
        num_workers = config.get('val_num_workers', 0) # Fewer workers often okay for validation

    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if device=='cuda' else False, # Pin memory if using GPU
        # persistent_workers=True if num_workers > 0 else False # Can speed up epoch start
    )
    
    # Return dataset instance as well, it contains useful info like bounds, K matrix etc.
    return loader, dataset 