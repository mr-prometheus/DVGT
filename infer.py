import torch
import os
from dvgt.models.dvgt import DVGT
from dvgt.utils.load_fn import load_and_preprocess_images_square
from iopath.common.file_io import g_pathmgr

checkpoint_path = 'models/DVGT/open_ckpt.pt'

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
model = DVGT()
with g_pathmgr.open(checkpoint_path, "rb") as f:
    checkpoint = torch.load(f, map_location="cpu")
model.load_state_dict(checkpoint)
model = model.to(device).eval()

# Load and preprocess example images
image_dir = '/home/de575594/Deepan/CV/geolocalization/vggt-long/datasets/bdd_dataset_day/val/extracted_frames/frames_4s/b1c9c847-3bda4659'

# Get all image paths and sort them by frame number
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]

# Sort by frame number extracted from filename (e.g., "b1c9c847-3bda4659_frame_0000_t0.00s.jpg")
def get_frame_number(filename):
    try:
        # Extract the frame number from pattern like "_frame_0000_"
        parts = filename.split('_frame_')
        if len(parts) > 1:
            frame_num = int(parts[1].split('_')[0])
            return frame_num
    except (ValueError, IndexError):
        pass
    return 0

image_files.sort(key=get_frame_number)

# Select specific frame range (adjust as needed)
start_frame = 0
end_frame = 10  # Adjust based on how many frames you want to process

# Filter files by frame number
selected_files = []
for f in image_files:
    frame_num = get_frame_number(f)
    if start_frame <= frame_num <= end_frame:
        selected_files.append(f)

# Build full paths
image_paths = [os.path.join(image_dir, f) for f in selected_files]

print(f"Found {len(image_paths)} images to process:")
for p in image_paths:
    print(f"  - {os.path.basename(p)}")

if len(image_paths) == 0:
    raise ValueError(f"No images found in {image_dir}")

# Load and preprocess images using the square preprocessing function
images, original_coords = load_and_preprocess_images_square(image_paths, target_size=512)

# Reshape to match expected input format: (B, T, V, C, H, W)
# Since you have single-view frames, V=1
# B=1 (batch), T=number of frames, V=1 (single view per frame)
T = images.shape[0]  # Number of frames
V = 1  # Single view per frame
C, H, W = images.shape[1], images.shape[2], images.shape[3]

images = images.view(1, T, V, C, H, W)
print(f"Input tensor shape: {images.shape}")  # Should be (1, T, 1, 3, 512, 512)

images = images.to(device)

with torch.no_grad():
    with torch.amp.autocast(device, dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

# Print prediction keys and shapes
print("\nPrediction outputs:")
if isinstance(predictions, dict):
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
else:
    print(f"  Output type: {type(predictions)}")
    if isinstance(predictions, torch.Tensor):
        print(f"  Output shape: {predictions.shape}")