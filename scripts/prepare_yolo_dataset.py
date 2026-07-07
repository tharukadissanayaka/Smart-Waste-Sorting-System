#!/usr/bin/env python3
"""
Script to prepare and format Kaggle Garbage Classification v2 and TACO datasets
into the YOLOv8 object detection format.

Target classes:
0: Plastic
1: Paper (includes Cardboard)
2: Glass
3: Metal (includes Metal, Trash, Organic, and other non-recyclable items)
"""

import argparse
import json
import random
import shutil
from pathlib import Path
import cv2
import yaml

# Default relative paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Format and split datasets into YOLO format."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw downloads."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save structured YOLO dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split reproducibility."
    )
    return parser


def get_target_class_from_name(name: str) -> int:
    """
    Map a raw category name (from Kaggle or TACO) to target classes:
    0: Plastic
    1: Paper
    2: Glass
    3: Metal
    """
    name_lower = name.lower().strip()
    
    # 0. Plastic
    if any(k in name_lower for k in [
        "plastic", "glove", "straw", "film", "bag", "wrapper", "lid", "cup lids", 
        "utensils", "bottle cap", "styrofoam", "garbage bag"
    ]):
        return 0
        
    # 1. Paper
    if any(k in name_lower for k in [
        "paper", "cardboard", "corrugated", "carton", "magazine", "newspaper", "book"
    ]):
        return 1
        
    # 2. Glass
    if any(k in name_lower for k in [
        "glass", "jar"
    ]):
        return 2
        
    # 3. Metal (acts as Metal/Organic/General Waste bucket)
    # This matches the methodology to merge "Trash" & organic into Metal/Organic
    return 3


def process_kaggle_dataset(raw_kaggle_dir: Path) -> list[dict]:
    """
    Process classification-only Kaggle garbage classification dataset.
    Generates a full-image bounding box as a fallback.
    """
    samples = []
    if not raw_kaggle_dir.exists():
        print(f"[INFO] Kaggle raw directory not found at {raw_kaggle_dir}. Skipping Kaggle dataset.")
        return samples

    # If the Kaggle dataset was downloaded with subfolders like 'original', use that
    original_dir = raw_kaggle_dir / "original"
    if original_dir.exists():
        raw_kaggle_dir = original_dir

    print(f"Processing Kaggle Garbage Classification v2 dataset from {raw_kaggle_dir}...")
    # List subdirectories (representing classes)
    for class_path in raw_kaggle_dir.iterdir():
        if not class_path.is_dir():
            continue
        
        raw_class_name = class_path.name
        target_class = get_target_class_from_name(raw_class_name)
        
        # Log mapping details
        print(f"  Kaggle raw class '{raw_class_name}' -> Target Class {target_class}")
        
        # Read images
        for img_path in class_path.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                # Bounding box is full image fallback: [class_id, x_center, y_center, w, h] normalized
                # We use 0.5, 0.5, 1.0, 1.0 (entire image is the bounding box)
                samples.append({
                    "image_path": img_path,
                    "labels": [[target_class, 0.5, 0.5, 1.0, 1.0]],
                    "source": "kaggle"
                })
                
    print(f"Loaded {len(samples)} samples from Kaggle dataset.")
    return samples


def process_taco_dataset(raw_taco_dir: Path) -> list[dict]:
    """Process TACO object detection dataset from COCO format annotations."""
    samples = []
    annotations_path = raw_taco_dir / "annotations.json"
    taco_images_dir = raw_taco_dir / "images"

    if not annotations_path.exists() or not taco_images_dir.exists():
        print(f"[INFO] TACO raw files not found at {raw_taco_dir}. Skipping TACO dataset.")
        return samples

    print("Processing TACO dataset...")
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Map COCO categories to our target classes
    category_mapping = {}
    for cat in coco_data.get("categories", []):
        cat_id = cat["id"]
        cat_name = cat["name"]
        target_class = get_target_class_from_name(cat_name)
        category_mapping[cat_id] = target_class
        
    print(f"Mapped {len(category_mapping)} TACO categories to target classes.")

    # Group annotations by image_id
    image_annotations = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    # Process images
    missing_images = 0
    for img_info in coco_data.get("images", []):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = taco_images_dir / file_name
        
        if not img_path.exists():
            missing_images += 1
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]
        
        # Retrieve annotations
        anns = image_annotations.get(img_id, [])
        labels = []
        
        for ann in anns:
            cat_id = ann["category_id"]
            target_class = category_mapping[cat_id]
            
            # COCO bbox format: [x_min, y_min, width, height]
            bbox = ann["bbox"]
            x_min, y_min, w, h = bbox
            
            # Convert to YOLO normalized format: [x_center, y_center, w, h]
            x_center = x_min + w / 2.0
            y_center = y_min + h / 2.0
            
            # Normalize by image dimensions
            x_center_norm = x_center / img_w
            y_center_norm = y_center / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            # Clip to [0, 1] to prevent invalid coordinates
            x_center_norm = max(0.0, min(1.0, x_center_norm))
            y_center_norm = max(0.0, min(1.0, y_center_norm))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))
            
            labels.append([target_class, x_center_norm, y_center_norm, w_norm, h_norm])
            
        # If the image has no annotations, we can still include it as background
        samples.append({
            "image_path": img_path,
            "labels": labels,
            "source": "taco"
        })

    if missing_images > 0:
        print(f"  [INFO] Missing {missing_images} TACO images (expected if using sample-limit download).")
    print(f"Loaded {len(samples)} samples from TACO dataset.")
    return samples


def main():
    parser = setup_args()
    args = parser.parse_args()
    
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    
    print(f"Preparing dataset from raw data: {raw_dir}")
    print(f"Target YOLO dataset directory: {output_dir}")
    
    # 1. Process Raw Data
    kaggle_samples = process_kaggle_dataset(raw_dir / "kaggle")
    taco_samples = process_taco_dataset(raw_dir / "taco")
    
    all_samples = kaggle_samples + taco_samples
    if not all_samples:
        print("No raw samples found. Please run scripts/download_data.py first.")
        return

    # 2. Split dataset: 70% Train, 20% Val, 10% Test
    # Set seed for reproducibility
    random.seed(args.seed)
    random.shuffle(all_samples)
    
    total = len(all_samples)
    train_end = int(total * 0.7)
    val_end = int(total * 0.9)
    
    splits = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:]
    }
    
    # 3. Create Directories
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    
    # Clean previous splits to avoid leftovers
    for split in ["train", "val", "test"]:
        shutil.rmtree(images_out / split, ignore_errors=True)
        shutil.rmtree(labels_out / split, ignore_errors=True)
        
        (images_out / split).mkdir(parents=True, exist_ok=True)
        (labels_out / split).mkdir(parents=True, exist_ok=True)

    # 4. Copy/Write splits
    print("Writing images and labels splits...")
    stats = {
        "train": {"total": 0, "kaggle": 0, "taco": 0, "classes": {0:0, 1:0, 2:0, 3:0}},
        "val": {"total": 0, "kaggle": 0, "taco": 0, "classes": {0:0, 1:0, 2:0, 3:0}},
        "test": {"total": 0, "kaggle": 0, "taco": 0, "classes": {0:0, 1:0, 2:0, 3:0}}
    }
    
    for split_name, split_data in splits.items():
        for sample in split_data:
            img_path = sample["image_path"]
            labels = sample["labels"]
            source = sample["source"]
            
            # Destination filenames
            # Prepend source to avoid conflicts if filenames are same
            dest_name = f"{source}_{img_path.name}"
            dest_img_path = images_out / split_name / dest_name
            dest_label_path = labels_out / split_name / f"{Path(dest_name).stem}.txt"
            
            # Copy Image
            shutil.copy2(img_path, dest_img_path)
            
            # Write Label file (YOLO format)
            with open(dest_label_path, "w") as lf:
                for lbl in labels:
                    class_id, x, y, w, h = lbl
                    lf.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    # Increment stats
                    stats[split_name]["classes"][class_id] += 1
            
            stats[split_name]["total"] += 1
            stats[split_name][source] += 1

    # 5. Generate data.yaml in Ultralytics format
    data_yaml = {
        "path": str(output_dir.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "Plastic",
            1: "Paper",
            2: "Glass",
            3: "Metal"
        }
    }
    
    with open(output_dir / "data.yaml", "w") as yf:
        yaml.safe_dump(data_yaml, yf, default_flow_style=False)
        
    print("\n" + "="*50)
    print("DATASET PREPARATION STATISTICS")
    print("="*50)
    for split_name, data in stats.items():
        print(f"Split: {split_name.upper()}")
        print(f"  Total Images: {data['total']} (Kaggle: {data['kaggle']}, TACO: {data['taco']})")
        print(f"  Class instances:")
        print(f"    Plastic (0): {data['classes'][0]}")
        print(f"    Paper (1)  : {data['classes'][1]}")
        print(f"    Glass (2)  : {data['classes'][2]}")
        print(f"    Metal (3)  : {data['classes'][3]}")
    print("="*50)
    print("Dataset preparation complete! data.yaml file created.")


if __name__ == "__main__":
    main()
