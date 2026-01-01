import os

from huggingface_hub import snapshot_download

# Download the model
print("Starting download of DVGT checkpoint...")
snapshot_download(
    repo_id="RainyNight/DVGT",
    local_dir="./models/DVGT",
    resume_download=True,  # Resume if interrupted
    max_workers=4,  # Parallel downloads
)

print("Download complete!")