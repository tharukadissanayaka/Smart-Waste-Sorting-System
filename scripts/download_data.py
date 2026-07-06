#!/usr/bin/env python3
"""
Script to download and prepare raw data sources for the Smart Waste Sorting System.
Supported datasets:
1. Kaggle "Garbage Classification v2" (sumn2u/garbage-classification-v2)
2. TACO (Trash Annotations in Context)

Instructions for setting up Kaggle API:
1. Go to Kaggle (https://www.kaggle.com), log in, and navigate to your account settings.
2. Click "Create New API Token" to download a file named `kaggle.json`.
3. Move `kaggle.json` to the appropriate directory:
   - Windows: C:\\Users\\<Your-Username>\\.kaggle\\kaggle.json
   - Linux/macOS: ~/.kaggle/kaggle.json
4. Make sure permissions are secure: `chmod 600 ~/.kaggle/kaggle.json` (Linux/macOS).
5. Alternatively, set environment variables KAGGLE_USERNAME and KAGGLE_KEY.
"""

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import requests

# Default relative paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download/prepare raw Kaggle and TACO datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to save raw downloads."
    )
    parser.add_argument(
        "--taco-limit",
        type=int,
        default=10,
        help="Limit number of TACO images to download (use 0 or -1 for all). Default is 10 for quick testing."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic/dummy data instead of downloading from APIs. Useful for testing without credentials."
    )
    return parser


def check_kaggle_credentials() -> bool:
    """Check if Kaggle credentials are set up."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_home = Path.home() / ".kaggle"
    kaggle_json = kaggle_home / "kaggle.json"
    return kaggle_json.exists()


def generate_synthetic_data(output_dir: Path):
    """Generate synthetic dummy images and metadata for dry-running the pipeline."""
    print("Generating synthetic dummy dataset for dry-run...")
    
    # Generate Kaggle dummy data
    # 10 categories
    kaggle_classes = [
        "Battery", "Biological", "Cardboard", "Clothes",
        "Glass", "Metal", "Paper", "Plastic", "Shoes", "Trash"
    ]
    kaggle_dir = output_dir / "kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 images per category
    print("Generating Kaggle mock classification images...")
    for cls in kaggle_classes:
        cls_dir = kaggle_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            img_path = cls_dir / f"mock_{cls.lower()}_{i}.jpg"
            # Create a simple colored square image
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            # Fill with different colors
            img[:] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
            # Draw class name text
            cv2.putText(img, cls[:3], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(str(img_path), img)

    # Generate TACO dummy data
    taco_dir = output_dir / "taco"
    taco_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fake annotations.json matching COCO format
    taco_images = []
    taco_annotations = []
    
    # Define a few mock categories
    taco_categories = [
        {"id": 0, "name": "Aluminium foil", "supercategory": "Aluminium"},
        {"id": 3, "name": "Bottle cap", "supercategory": "Plastic"},
        {"id": 4, "name": "Clear plastic bottle", "supercategory": "Plastic"},
        {"id": 5, "name": "Glass bottle", "supercategory": "Glass"},
        {"id": 7, "name": "Soda can", "supercategory": "Aluminium"},
        {"id": 13, "name": "Paper bag", "supercategory": "Paper"},
        {"id": 16, "name": "Cardboard", "supercategory": "Paper"},
        {"id": 59, "name": "Cigarette", "supercategory": "Trash"}
    ]
    
    # Create mock images folder
    taco_img_dir = taco_dir / "images"
    taco_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 5 mock images with bounding boxes
    print("Generating TACO mock detection images & annotations...")
    for i in range(5):
        # Create different subfolders like TACO does
        batch_folder = f"batch_{i // 2 + 1}"
        file_path_rel = f"{batch_folder}/taco_mock_{i}.jpg"
        img_path = taco_img_dir / batch_folder / f"taco_mock_{i}.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy image
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:] = [100, i * 40 + 50, 100]
        cv2.imwrite(str(img_path), img)
        
        taco_images.append({
            "id": i,
            "file_name": file_path_rel,
            "width": 256,
            "height": 256,
            "flickr_url": ""
        })
        
        # Add 1-2 annotations per image
        category_info = taco_categories[i % len(taco_categories)]
        taco_annotations.append({
            "id": i * 10,
            "image_id": i,
            "category_id": category_info["id"],
            "bbox": [50.0, 50.0, 100.0, 100.0],  # [x_min, y_min, width, height]
            "area": 10000.0,
            "iscrowd": 0
        })
        
    annotations_json = {
        "images": taco_images,
        "annotations": taco_annotations,
        "categories": taco_categories
    }
    
    with open(taco_dir / "annotations.json", "w") as f:
        json.dump(annotations_json, f, indent=4)
        
    print("Synthetic mock dataset generated successfully!")


def download_kaggle(kaggle_dir: Path):
    """Download Kaggle dataset using the kaggle package API."""
    if not check_kaggle_credentials():
        raise ValueError(
            "Kaggle credentials not found. Please place `kaggle.json` in ~/.kaggle/ "
            "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
    
    # Import kaggle here to avoid import error during dry-run if not installed
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "The `kaggle` library is not installed. Please run `pip install kaggle` first."
        )
    
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    dataset_name = "sumn2u/garbage-classification-v2"
    print(f"Downloading {dataset_name} from Kaggle (this may take a few minutes)...")
    
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    # This downloads and extracts files
    api.dataset_download_files(dataset_name, path=str(kaggle_dir), unzip=True)
    print("Kaggle dataset downloaded and extracted successfully.")


def download_taco(taco_dir: Path, limit: int):
    """Download TACO annotations and a subset of images from Flickr."""
    print("Downloading TACO dataset annotations...")
    taco_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_url = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
    annotations_path = taco_dir / "annotations.json"
    
    # Fetch annotations.json
    try:
        response = requests.get(annotations_url, timeout=30)
        response.raise_for_status()
        with open(annotations_path, "w") as f:
            f.write(response.text)
        print("TACO annotations saved.")
    except Exception as e:
        print(f"Error downloading annotations: {e}")
        return

    # Load annotations to get image URLs
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    images = annotations.get("images", [])
    total_images = len(images)
    print(f"TACO dataset has {total_images} total images defined.")
    
    # Limit number of images to download
    if limit > 0:
        images_to_download = images[:limit]
        print(f"Downloading limited subset of {limit} images...")
    else:
        images_to_download = images
        print(f"Downloading all {total_images} images (warning: this will take time & space)...")

    images_dir = taco_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    for idx, img_info in enumerate(images_to_download):
        file_name = img_info["file_name"]
        flickr_url = img_info.get("flickr_url") or img_info.get("flickr_640_url")
        
        if not flickr_url:
            print(f"Skipping {file_name}: No download URL available.")
            continue
            
        img_out_path = images_dir / file_name
        img_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if img_out_path.exists():
            print(f"[{idx+1}/{len(images_to_download)}] Already exists: {file_name}")
            downloaded_count += 1
            continue

        try:
            print(f"[{idx+1}/{len(images_to_download)}] Downloading {flickr_url} -> {file_name}")
            # Request with headers to avoid user-agent blocking
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            img_response = requests.get(flickr_url, headers=headers, timeout=20)
            img_response.raise_for_status()
            
            with open(img_out_path, "wb") as img_file:
                img_file.write(img_response.content)
            downloaded_count += 1
        except Exception as e:
            print(f"Error downloading {file_name} from {flickr_url}: {e}")
            
    print(f"Successfully downloaded {downloaded_count} of {len(images_to_download)} TACO images.")


def main():
    parser = setup_args()
    args = parser.parse_args()
    
    output_dir = args.output_dir.resolve()
    print(f"Raw dataset output directory: {output_dir}")
    
    if args.dry_run:
        generate_synthetic_data(output_dir)
        print("Data download step complete (Dry-Run Mode).")
        return

    # Real download workflow
    # 1. Kaggle
    kaggle_dir = output_dir / "kaggle"
    if check_kaggle_credentials():
        try:
            download_kaggle(kaggle_dir)
        except Exception as e:
            print(f"\n[WARNING] Kaggle download failed: {e}")
            print("Falling back to generating synthetic Kaggle data...")
            generate_synthetic_data(output_dir)
            return
    else:
        print("\n" + "="*80)
        print("WARNING: Kaggle credentials not set up or found.")
        print("To download the real Kaggle garbage classification dataset, please:")
        print("1. Download your API key `kaggle.json` from Kaggle settings.")
        print("2. Save it to ~/.kaggle/kaggle.json (or specify KAGGLE_USERNAME/KAGGLE_KEY env vars).")
        print("\nFor now, we are falling back to generating synthetic/dummy data so you can test the pipeline.")
        print("="*80 + "\n")
        generate_synthetic_data(output_dir)
        return

    # 2. TACO
    taco_dir = output_dir / "taco"
    try:
        download_taco(taco_dir, args.taco_limit)
    except Exception as e:
        print(f"Error occurred during TACO download: {e}")


if __name__ == "__main__":
    main()
