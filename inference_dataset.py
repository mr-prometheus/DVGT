import torch
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from einops import rearrange
from pathlib import Path
import traceback

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


def save_point_cloud_npz(points, colors, poses, filename):
    """Save point cloud and poses to NPZ file."""
    np.savez(
        filename,
        points=points,
        colors=colors,
        poses=poses
    )


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


def process_single_video(model, device, dtype, video_id, frame_dir, output_base_dir, 
                         frame_type, is_masked, conf_threshold=25.0, max_depth=-1):
    """
    Process a single video for a specific frame type and masked variation.
    
    Args:
        model: DVGT model
        device: torch device
        dtype: torch dtype
        video_id: Video identifier (e.g., 'b1c9c847-3bda4659')
        frame_dir: Directory containing frames for this video
        output_base_dir: Base output directory
        frame_type: Frame interval type (e.g., '4s', '5s', '6s', '7s')
        is_masked: Whether this is the masked variant
        conf_threshold: Confidence threshold for filtering
        max_depth: Maximum depth for filtering
    
    Returns:
        True if successful, False otherwise
    """
    # Determine frame range based on frame type
    # Assuming frames are extracted at 1fps, so 4s = frames 0-24
    frame_ranges = {
        '4s': (0, 24),  # 4 seconds at 1fps = 4 frames, but model can use up to 24
        '5s': (0, 24),
        '6s': (0, 24),
        '7s': (0, 24),
    }
    
    start_frame, end_frame = frame_ranges.get(frame_type, (0, 24))
    
    # Create output directory structure
    mask_suffix = '_masked' if is_masked else ''
    output_dir = os.path.join(output_base_dir, f'frames_{frame_type}{mask_suffix}', video_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory with proper frame structure
    temp_dir = tempfile.mkdtemp(prefix=f'dvgt_{video_id}_')
    
    try:
        # Reorganize images
        num_frames = reorganize_images_to_frame_dirs(
            frame_dir, temp_dir, 
            start_frame=start_frame, 
            end_frame=end_frame
        )
        
        if num_frames == 0:
            print(f"  ⚠️  No frames found for {video_id} in range [{start_frame}, {end_frame}]")
            return False
        
        # Load and preprocess images
        images = load_and_preprocess_images(temp_dir).to(device)
        
        # Run inference
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=dtype):
                predictions = model(images)
        
        # Process predictions
        points, colors, poses = process_predictions(
            predictions, 
            conf_threshold=conf_threshold,
            max_depth=max_depth
        )
        
        # Save outputs
        ply_path = os.path.join(output_dir, 'point_cloud.ply')
        npz_path = os.path.join(output_dir, 'point_cloud.npz')
        
        save_point_cloud_ply(points, colors, ply_path)
        save_point_cloud_npz(points, colors, poses, npz_path)
        
        print(f"  ✓ Saved {points.shape[0]} points to {output_dir}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {video_id}: {str(e)}")
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    # ============== CONFIGURATION ==============
    
    # Path to metadata CSV
    metadata_csv = '/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/metadata_test.csv'
    
    # Base directory for extracted frames
    extracted_frames_base = '/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/extracted_frames'
    
    # Output directory
    output_base_dir = './outputs/point_clouds'
    
    # Checkpoint path
    checkpoint_path = 'models/DVGT/open_ckpt.pt'
    
    # Frame types to process
    frame_types = ['4s', '5s', '6s', '7s']
    
    # Whether to process masked variants
    process_masked = True
    
    # Number of videos to process (set to None for all)
    max_videos = 50
    
    # Filtering settings
    conf_threshold = 25.0
    max_depth = -1
    
    # ============================================
    
    print("="*80)
    print("DVGT BATCH INFERENCE")
    print("="*80)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"\n📍 Device: {device}, dtype: {dtype}")
    
    # Load metadata
    print(f"\n📊 Loading metadata from: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    
    # Extract video IDs from filenames
    df['video_id'] = df['filename'].str.replace('.mov', '', regex=False)
    
    # Limit number of videos if specified
    if max_videos is not None:
        df = df.head(max_videos)
        print(f"   Processing first {len(df)} videos")
    else:
        print(f"   Processing all {len(df)} videos")
    
    # Load model
    print(f"\n🤖 Loading model from: {checkpoint_path}")
    model = DVGT()
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()
    print("   ✓ Model loaded successfully!")
    
    # Prepare processing tasks
    tasks = []
    for _, row in df.iterrows():
        video_id = row['video_id']
        
        for frame_type in frame_types:
            # Process unmasked version
            frame_dir = os.path.join(extracted_frames_base, f'frames_{frame_type}', video_id)
            if os.path.exists(frame_dir):
                tasks.append((video_id, frame_dir, frame_type, False))
            
            # Process masked version if enabled
            if process_masked:
                masked_frame_dir = os.path.join(extracted_frames_base, f'frames_{frame_type}_masked', video_id)
                if os.path.exists(masked_frame_dir):
                    tasks.append((video_id, masked_frame_dir, frame_type, True))
    
    print(f"\n📋 Total tasks to process: {len(tasks)}")
    print(f"   Videos: {len(df)}")
    print(f"   Frame types: {', '.join(frame_types)}")
    print(f"   Variations per video: {len(frame_types) * (2 if process_masked else 1)}")
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process all tasks
    print(f"\n{'='*80}")
    print("PROCESSING")
    print("="*80)
    
    success_count = 0
    fail_count = 0
    
    for idx, (video_id, frame_dir, frame_type, is_masked) in enumerate(tasks, 1):
        mask_suffix = '_masked' if is_masked else ''
        print(f"\n[{idx}/{len(tasks)}] Processing: {video_id} (frames_{frame_type}{mask_suffix})")
        
        success = process_single_video(
            model=model,
            device=device,
            dtype=dtype,
            video_id=video_id,
            frame_dir=frame_dir,
            output_base_dir=output_base_dir,
            frame_type=frame_type,
            is_masked=is_masked,
            conf_threshold=conf_threshold,
            max_depth=max_depth
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Final summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE!")
    print("="*80)
    print(f"✓ Successful: {success_count}/{len(tasks)}")
    print(f"✗ Failed: {fail_count}/{len(tasks)}")
    print(f"\n📂 Output directory: {output_base_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_base_dir}/")
    print(f"    ├── frames_4s/")
    print(f"    │   └── <video_id>/")
    print(f"    │       ├── point_cloud.ply")
    print(f"    │       └── point_cloud.npz")
    print(f"    ├── frames_4s_masked/")
    print(f"    ├── frames_5s/")
    print(f"    ├── frames_5s_masked/")
    print(f"    └── ...")


if __name__ == "__main__":
    main()