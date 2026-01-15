import torch
import os
import shutil
import tempfile
import numpy as np
from einops import rearrange

from dvgt.models.dvgt import DVGT
from dvgt.utils.load_fn import load_and_preprocess_images
from iopath.common.file_io import g_pathmgr
from dvgt.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n
from dvgt.utils.pose_enc import pose_encoding_to_ego_pose


def get_frame_number(filename):
    """Extract frame number from filename like 'b1c9c847-3bda4659_frame_0000_t0.00s.jpg'"""
    try:
        parts = filename.split('_frame_')
        if len(parts) > 1:
            frame_num = int(parts[1].split('_')[0])
            return frame_num
    except (ValueError, IndexError):
        pass
    return -1


def reorganize_images_to_frame_dirs(src_dir, dst_dir, start_frame=None, end_frame=None):
    """
    Reorganize flat image files into frame_x subdirectories.
    
    Args:
        src_dir: Source directory with flat image files
        dst_dir: Destination directory to create frame_x structure
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (inclusive)
    
    Returns:
        Number of frames created
    """
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_exts)]
    
    # Sort by frame number
    image_files.sort(key=get_frame_number)
    
    # Filter by frame range
    if start_frame is not None or end_frame is not None:
        filtered_files = []
        for f in image_files:
            frame_num = get_frame_number(f)
            if frame_num < 0:
                continue
            if start_frame is not None and frame_num < start_frame:
                continue
            if end_frame is not None and frame_num > end_frame:
                continue
            filtered_files.append(f)
        image_files = filtered_files
    
    # Create frame directories and copy images
    frame_count = 0
    for f in image_files:
        frame_num = get_frame_number(f)
        if frame_num < 0:
            continue
            
        frame_dir = os.path.join(dst_dir, f'frame_{frame_num}')
        os.makedirs(frame_dir, exist_ok=True)
        
        # Copy image to frame directory as a view (e.g., CAM_F.jpg)
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(frame_dir, 'CAM_F.jpg')  # Single view
        shutil.copy(src_path, dst_path)
        frame_count += 1
    
    return frame_count


def save_point_cloud_ply(points, colors, filename):
    """
    Save point cloud to PLY file format.
    
    Args:
        points: numpy array of shape (N, 3)
        colors: numpy array of shape (N, 3) with values 0-255
    """
    num_points = points.shape[0]
    
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = colors[i].astype(np.uint8)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    print(f"Saved point cloud with {num_points} points to {filename}")


def save_point_cloud_npz(points, colors, poses, filename):
    """Save point cloud and poses to NPZ file."""
    np.savez(
        filename,
        points=points,
        colors=colors,
        poses=poses
    )
    print(f"Saved point cloud data to {filename}")


def process_predictions(predictions, conf_threshold=25.0, max_depth=-1):
    """
    Process model predictions to extract point cloud.
    
    Args:
        predictions: Model output dictionary
        conf_threshold: Percentage of low-confidence points to filter out
        max_depth: Maximum depth to keep (-1 for no limit)
    
    Returns:
        points, colors, poses
    """
    B, T, V, H, W, _ = predictions['world_points'].shape
    device = predictions['world_points'].device
    
    # Get ego poses
    pred_ego_n_to_ego_0 = pose_encoding_to_ego_pose(predictions['ego_pose_enc'])
    
    # Get points and confidence
    pred_points = predictions['world_points'][0].cpu().numpy()  # T, V, H, W, 3
    pred_points_conf = predictions['world_points_conf'][0].cpu().numpy()  # T, V, H, W
    pred_ego_poses = pred_ego_n_to_ego_0[0].cpu().numpy()  # T, 3, 4
    
    # Get images for colors
    images = rearrange(predictions['images'][0].cpu().numpy(), 't v c h w -> t v h w c') * 255
    images = images.astype(np.uint8)
    
    # Create confidence mask
    combined_mask = np.ones((T, V, H, W), dtype=bool)
    
    if conf_threshold > 0:
        cutoff_value = np.percentile(pred_points_conf, conf_threshold)
        conf_mask = pred_points_conf >= cutoff_value
        combined_mask &= conf_mask
    
    # Apply max depth filter
    if max_depth > 0:
        depth = np.linalg.norm(pred_points, axis=-1)
        depth_mask = depth <= max_depth
        combined_mask &= depth_mask
    
    # Flatten and filter
    mask_flat = combined_mask.reshape(-1)
    points_flat = pred_points.reshape(-1, 3)
    colors_flat = images.reshape(-1, 3)
    
    points_filtered = points_flat[mask_flat]
    colors_filtered = colors_flat[mask_flat]
    
    # Center the point cloud
    if points_filtered.shape[0] > 0:
        center = np.mean(points_filtered, axis=0)
        points_filtered = points_filtered - center
        pred_ego_poses[..., -1] -= center
    
    return points_filtered, colors_filtered, pred_ego_poses


def main():
    # ============== CONFIGURATION ==============
    # Path to your image directory (with flat image files)
    image_dir = '/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/extracted_frames/frames_5s_masked/b1c9c847-3bda4659'
    
    # Checkpoint path
    checkpoint_path = 'models/DVGT/open_ckpt.pt'
    
    # Frame range (adjust based on your data)
    start_frame = 0
    end_frame = 24  # Model supports up to 24 frames
    
    # Output settings
    output_dir = './outputs'
    output_prefix = 'point_cloud'
    
    # Filtering settings
    conf_threshold = 25.0  # Filter out bottom 25% low-confidence points
    max_depth = -1  # Set to positive value (e.g., 50) to limit depth in meters
    
    # ============================================
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load model
    print("Loading model...")
    model = DVGT()
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()
    print("Model loaded successfully!")
    
    # Create temporary directory with proper frame structure
    print(f"\nReorganizing images from: {image_dir}")
    temp_dir = tempfile.mkdtemp(prefix='dvgt_frames_')
    
    try:
        num_frames = reorganize_images_to_frame_dirs(
            image_dir, temp_dir, 
            start_frame=start_frame, 
            end_frame=end_frame
        )
        print(f"Reorganized {num_frames} frames into: {temp_dir}")
        
        if num_frames == 0:
            raise ValueError("No frames found in the specified range!")
        
        # List created frame directories
        frame_dirs = sorted(os.listdir(temp_dir))
        print(f"Frame directories: {frame_dirs[:5]}..." if len(frame_dirs) > 5 else f"Frame directories: {frame_dirs}")
        
        # Load and preprocess images
        print("\nLoading and preprocessing images...")
        images = load_and_preprocess_images(temp_dir).to(device)
        print(f"Input tensor shape: {images.shape}")  # Should be (1, T, V, C, H, W)
        
        # Run inference
        print("\nRunning inference...")
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=dtype):
                predictions = model(images)
        
        print("Inference complete!")
        
        # Print prediction keys
        print("\nPrediction outputs:")
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Process predictions to get point cloud
        print("\nProcessing predictions...")
        points, colors, poses = process_predictions(
            predictions, 
            conf_threshold=conf_threshold,
            max_depth=max_depth
        )
        
        print(f"Final point cloud: {points.shape[0]} points")
        
        # Save outputs
        ply_path = os.path.join(output_dir, f'{output_prefix}.ply')
        npz_path = os.path.join(output_dir, f'{output_prefix}.npz')
        
        save_point_cloud_ply(points, colors, ply_path)
        save_point_cloud_npz(points, colors, poses, npz_path)
        
        print(f"\n{'='*50}")
        print("INFERENCE COMPLETE!")
        print(f"{'='*50}")
        print(f"Point cloud saved to:")
        print(f"  - PLY: {ply_path}")
        print(f"  - NPZ: {npz_path}")
        print(f"\nYou can view the PLY file in:")
        print(f"  - MeshLab")
        print(f"  - CloudCompare")
        print(f"  - Open3D")
        print(f"  - Blender")
        
    finally:
        # Clean up temporary directory
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()