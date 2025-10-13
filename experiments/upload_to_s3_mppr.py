#!/usr/bin/env python3
"""
Upload activation files to RunPod S3 using MPPR's built-in upload method.
"""
import sys
sys.path.insert(0, "/root/repeng")

from pathlib import Path
from dotenv import load_dotenv
from mppr import MContext

# Load RunPod S3 credentials from .env
load_dotenv("/root/repeng/.env")

# Find all activation files
repo_root = Path("/root/repeng")
output_dir = repo_root / "output" / "comparison"

if not output_dir.exists():
    print("No output directory found")
    sys.exit(1)

# Find all models with activations
activation_files = []
for model_dir in output_dir.iterdir():
    if not model_dir.is_dir():
        continue

    activation_file = model_dir / "activations_results" / "value.pickle"
    if activation_file.exists():
        activation_files.append((model_dir.name, activation_file))

if not activation_files:
    print("No activation files found")
    sys.exit(1)

print(f"Found {len(activation_files)} activation file(s) to upload:")
for model_name, filepath in activation_files:
    size_gb = filepath.stat().st_size / (1024**3)
    print(f"  - {model_name}: {size_gb:.2f} GB")

# Upload each file using MPPR
for model_name, filepath in activation_files:
    print(f"\nUploading {model_name}...")

    # Create a temporary MContext to use upload
    mcontext = MContext(filepath.parent.parent)

    # Create an MDict from the pickle file
    activations_dict = mcontext.create({"activations": "activations"})

    # Load the existing activations
    print(f"  Loading from: {filepath}")
    import pickle
    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    # Upload to S3
    s3_path = f"s3://mats/datasets/activations/{model_name}_v1.pickle"
    print(f"  Uploading to: {s3_path}")

    try:
        # Create a single-item dict and upload
        upload_dict = mcontext.create({"data": results})
        upload_dict.upload(s3_path, to="pickle")
        print(f"  ✓ Successfully uploaded {model_name}")
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")

print("\nUpload complete!")
