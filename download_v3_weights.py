#!/usr/bin/env python3
"""
Download DINOv3 weights from HuggingFace.
Run this script BEFORE running inference.

Uses the correct DINOv3 model: facebook/dinov3-vitl16-pretrain-lvd1689m
"""

import os
import argparse
from pathlib import Path


def download_dinov3_weights(hf_token: str = None, output_dir: str = None):
    """Download DINOv3 ViT-L/16 weights from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download, login
        
        if hf_token:
            login(token=hf_token)
            print("Logged in to HuggingFace")
        
        if output_dir is None:
            output_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        
        # DINOv3 ViT-Large model - the CORRECT model
        repo_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        
        print(f"Downloading DINOv3 ViT-L/16 weights from: {repo_id}")
        print("This is the correct DINOv3 model (NOT DINOv2)!")
        
        # Download the model weights
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            cache_dir=output_dir,
            token=hf_token
        )
        
        print(f"Downloaded to: {downloaded_path}")
        
        # Convert safetensors to pytorch format and save to expected location
        target_path = os.path.join(output_dir, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
        
        if not os.path.exists(target_path):
            print("Converting safetensors to PyTorch format...")
            try:
                from safetensors.torch import load_file
                import torch
                
                state_dict = load_file(downloaded_path)
                torch.save(state_dict, target_path)
                print(f"Converted and saved to: {target_path}")
            except ImportError:
                print("safetensors not installed. Install with: pip install safetensors")
                print(f"Alternatively, you can use the safetensors file directly at: {downloaded_path}")
                return downloaded_path
        else:
            print(f"Target file already exists: {target_path}")
        
        return target_path
        
    except ImportError as e:
        print(f"Required package not installed: {e}")
        print("Install with: pip install huggingface_hub safetensors")
        return None
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_with_transformers(output_dir: str = None):
    """Alternative: Download using transformers library."""
    try:
        import torch
        from transformers import AutoModel
        
        if output_dir is None:
            output_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        
        repo_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        print(f"Downloading DINOv3 ViT-L/16 using transformers from: {repo_id}")
        
        # Load the model (this will download the weights)
        model = AutoModel.from_pretrained(repo_id)
        
        # Save state dict in PyTorch format
        target_path = os.path.join(output_dir, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
        torch.save(model.state_dict(), target_path)
        
        print(f"Saved to: {target_path}")
        return target_path
        
    except ImportError:
        print("transformers not installed. Install with: pip install transformers")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_manual_instructions():
    """Show manual download instructions."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
    target_file = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS FOR DINOv3")
    print("=" * 70)
    
    print("\nThe correct model is: facebook/dinov3-vitl16-pretrain-lvd1689m")
    print("NOT dinov2-large!")
    
    print("\nMethod 1: Using huggingface_hub (recommended)")
    print("-" * 50)
    print("pip install huggingface_hub safetensors")
    print("python download_dinov3_weights.py")
    
    print("\nMethod 2: Using transformers")
    print("-" * 50)
    print("pip install transformers")
    print("python download_dinov3_weights.py --method transformers")
    
    print("\nMethod 3: Direct download from HuggingFace")
    print("-" * 50)
    print("Visit: https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m")
    print("Download model.safetensors and convert to .pth")
    
    print(f"\nExpected output location: {cache_dir}/{target_file}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download DINOv3 ViT-L/16 weights")
    parser.add_argument("--hf_token", type=str, default=None, 
                        help="HuggingFace token for authentication (optional)")
    parser.add_argument("--method", type=str, default="huggingface_hub",
                        choices=["huggingface_hub", "transformers", "manual"],
                        help="Download method")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for weights")
    args = parser.parse_args()
    
    print("=" * 70)
    print("DINOv3 Weight Downloader")
    print("Model: facebook/dinov3-vitl16-pretrain-lvd1689m")
    print("=" * 70)
    
    if args.method == "huggingface_hub":
        result = download_dinov3_weights(args.hf_token, args.output_dir)
        if result:
            print(f"\nSuccess! DINOv3 weights saved to: {result}")
            print("\nYou can now run inference with:")
            print(f"  --dinov3_weights_path {result}")
        else:
            print("\nDownload failed. Showing manual instructions...")
            show_manual_instructions()
    elif args.method == "transformers":
        result = download_with_transformers(args.output_dir)
        if result:
            print(f"\nSuccess! DINOv3 weights saved to: {result}")
            print("\nYou can now run inference with:")
            print(f"  --dinov3_weights_path {result}")
        else:
            print("\nDownload failed. Showing manual instructions...")
            show_manual_instructions()
    else:
        show_manual_instructions()


if __name__ == "__main__":
    main()