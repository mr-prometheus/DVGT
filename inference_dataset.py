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
    try:
        parts = filename.split('frame_')
        if len(parts) > 1:
            frame_num = int(parts[1].split('_')[0])
            return frame_num
    except (ValueError, IndexError):
        pass
    return -1


def reorganize_images_to_frame_dirs(src_dir, dst_dir):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_exts)]
    image_files.sort(key=get_frame_number)
    
    frame_count = 0
    for f in image_files:
        frame_num = get_frame_number(f)
        if frame_num < 0:
            continue
            
        frame_dir = os.path.join(dst_dir, f'frame_{frame_num}')
        os.makedirs(frame_dir, exist_ok=True)
        
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(frame_dir, 'CAM_F.jpg')
        shutil.copy(src_path, dst_path)
        frame_count += 1
    
    return frame_count


def save_point_cloud_ply(points, colors, filename):
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
    np.savez(
        filename,
        points=points,
        colors=colors,
        poses=poses
    )


def process_predictions(predictions, conf_threshold=25.0, max_depth=-1):
    B, T, V, H, W, _ = predictions['world_points'].shape
    device = predictions['world_points'].device
    
    pred_ego_n_to_ego_0 = pose_encoding_to_ego_pose(predictions['ego_pose_enc'])
    
    pred_points = predictions['world_points'][0].cpu().numpy()
    pred_points_conf = predictions['world_points_conf'][0].cpu().numpy()
    pred_ego_poses = pred_ego_n_to_ego_0[0].cpu().numpy()
    
    images = rearrange(predictions['images'][0].cpu().numpy(), 't v c h w -> t v h w c') * 255
    images = images.astype(np.uint8)
    
    combined_mask = np.ones((T, V, H, W), dtype=bool)
    
    if conf_threshold > 0:
        cutoff_value = np.percentile(pred_points_conf, conf_threshold)
        conf_mask = pred_points_conf >= cutoff_value
        combined_mask &= conf_mask
    
    if max_depth > 0:
        depth = np.linalg.norm(pred_points, axis=-1)
        depth_mask = depth <= max_depth
        combined_mask &= depth_mask
    
    mask_flat = combined_mask.reshape(-1)
    points_flat = pred_points.reshape(-1, 3)
    colors_flat = images.reshape(-1, 3)
    
    points_filtered = points_flat[mask_flat]
    colors_filtered = colors_flat[mask_flat]
    
    if points_filtered.shape[0] > 0:
        center = np.mean(points_filtered, axis=0)
        points_filtered = points_filtered - center
        pred_ego_poses[..., -1] -= center
    
    return points_filtered, colors_filtered, pred_ego_poses


def process_single_clip(model, device, dtype, video_id, clip_id, frame_dir, output_dir,
                        conf_threshold=25.0, max_depth=-1):
    temp_dir = tempfile.mkdtemp(prefix=f'dvgt_{video_id}_{clip_id}_')
    
    try:
        num_frames = reorganize_images_to_frame_dirs(frame_dir, temp_dir)
        
        if num_frames == 0:
            print(f"    ⚠️  No frames found for {video_id}/{clip_id}")
            return False
        
        images = load_and_preprocess_images(temp_dir).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=dtype):
                predictions = model(images)
        
        points, colors, poses = process_predictions(
            predictions, 
            conf_threshold=conf_threshold,
            max_depth=max_depth
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        ply_path = os.path.join(output_dir, 'point_cloud.ply')
        npz_path = os.path.join(output_dir, 'point_cloud.npz')
        
        save_point_cloud_ply(points, colors, ply_path)
        save_point_cloud_npz(points, colors, poses, npz_path)
        
        print(f"    ✓ Saved {points.shape[0]} points")
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {str(e)}")
        traceback.print_exc()
        return False
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def process_video_clips(model, device, dtype, video_id, video_dir, output_base_dir,
                        conf_threshold=25.0, max_depth=-1):
    clip_dirs = sorted([d for d in os.listdir(video_dir) if d.startswith('clip_')])
    
    if not clip_dirs:
        print(f"  ⚠️  No clips found for {video_id}")
        return 0, 0
    
    success_count = 0
    fail_count = 0
    
    for clip_id in clip_dirs:
        clip_path = os.path.join(video_dir, clip_id)
        
        original_dir = os.path.join(clip_path, 'original')
        masked_dir = os.path.join(clip_path, 'masked')
        
        if os.path.exists(original_dir):
            print(f"  [{clip_id}] Processing original...")
            output_dir = os.path.join(output_base_dir, video_id, clip_id, 'original')
            success = process_single_clip(
                model, device, dtype, video_id, clip_id, original_dir, output_dir,
                conf_threshold, max_depth
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        if os.path.exists(masked_dir):
            print(f"  [{clip_id}] Processing masked...")
            output_dir = os.path.join(output_base_dir, video_id, clip_id, 'masked')
            success = process_single_clip(
                model, device, dtype, video_id, clip_id, masked_dir, output_dir,
                conf_threshold, max_depth
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    return success_count, fail_count


def main():
    metadata_csv = '/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/metadata_clips.csv'
    extracted_clips_base = '/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/extracted_clips'
    output_base_dir = './outputs/point_clouds'
    checkpoint_path = 'models/DVGT/open_ckpt.pt'
    max_videos = 15
    conf_threshold = 25.0
    max_depth = -1
    
    print("="*80)
    print("DVGT BATCH INFERENCE - CLIP-BASED")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"\n📍 Device: {device}, dtype: {dtype}")
    
    print(f"\n📊 Loading metadata from: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    
    df['video_id'] = df['filename'].str.replace('.mov', '', regex=False)
    
    if max_videos is not None:
        df = df.head(max_videos)
        print(f"   Processing first {len(df)} videos")
    else:
        print(f"   Processing all {len(df)} videos")
    
    print(f"\n🤖 Loading model from: {checkpoint_path}")
    model = DVGT()
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()
    print("   ✓ Model loaded successfully!")
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("PROCESSING")
    print("="*80)
    
    total_success = 0
    total_fail = 0
    
    for idx, row in df.iterrows():
        video_id = row['video_id']
        num_clips = row.get('num_clips', 0)
        
        print(f"\n[{idx+1}/{len(df)}] Video: {video_id} ({num_clips} clips)")
        
        video_dir = os.path.join(extracted_clips_base, video_id)
        
        if not os.path.exists(video_dir):
            print(f"  ⚠️  Video directory not found: {video_dir}")
            continue
        
        success, fail = process_video_clips(
            model=model,
            device=device,
            dtype=dtype,
            video_id=video_id,
            video_dir=video_dir,
            output_base_dir=output_base_dir,
            conf_threshold=conf_threshold,
            max_depth=max_depth
        )
        
        total_success += success
        total_fail += fail
        print(f"  Video complete: {success} success, {fail} failed")
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE!")
    print("="*80)
    print(f"✓ Successful: {total_success}")
    print(f"✗ Failed: {total_fail}")
    print(f"\n📂 Output directory: {output_base_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_base_dir}/")
    print(f"    └── <video_id>/")
    print(f"        ├── clip_0000/")
    print(f"        │   ├── original/")
    print(f"        │   │   ├── point_cloud.ply")
    print(f"        │   │   └── point_cloud.npz")
    print(f"        │   └── masked/")
    print(f"        │       ├── point_cloud.ply")
    print(f"        │       └── point_cloud.npz")
    print(f"        ├── clip_0001/")
    print(f"        │   ├── original/")
    print(f"        │   └── masked/")
    print(f"        └── ...")


if __name__ == "__main__":
    main()