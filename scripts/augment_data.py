#!/usr/bin/env python3
"""
Script to apply Albumentations data augmentation to the training split of the YOLO dataset.
Applies: Random rotation, brightness/contrast jitter, and Gaussian noise.
"""

import argparse
import random
from pathlib import Path
import albumentations as A
import cv2

# Default relative paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply Albumentations augmentation to YOLO training split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Dataset root directory containing images/ and labels/."
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=2,
        help="Number of augmented images to generate per original training image."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    return parser


def load_yolo_labels(label_path: Path) -> tuple[list[list[float]], list[int]]:
    """Reads YOLO annotations from a .txt file."""
    bboxes = []
    class_labels = []
    
    if not label_path.exists():
        return bboxes, class_labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                # YOLO format: class_id, x_center, y_center, width, height
                coords = [float(x) for x in parts[1:]]
                
                # Clip coords to [0.0, 1.0] to satisfy Albumentations
                # x_center, y_center, w, h
                w = max(0.001, min(1.0, coords[2]))
                h = max(0.001, min(1.0, coords[3]))
                x = max(w / 2.0, min(1.0 - w / 2.0, coords[0]))
                y = max(h / 2.0, min(1.0 - h / 2.0, coords[1]))
                
                bboxes.append([x, y, w, h])
                class_labels.append(class_id)
                
    return bboxes, class_labels


def write_yolo_labels(label_path: Path, bboxes: list[list[float]], class_labels: list[int]):
    """Writes YOLO annotations to a .txt file."""
    with open(label_path, "w") as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x, y, w, h = bbox
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    parser = setup_args()
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    data_dir = args.data_dir.resolve()
    train_images_dir = data_dir / "images" / "train"
    train_labels_dir = data_dir / "labels" / "train"
    
    if not train_images_dir.exists() or not train_labels_dir.exists():
        print(f"Training split folders not found at {train_images_dir}. Run prepare_yolo_dataset.py first.")
        return

    print(f"Applying data augmentation on training set in: {train_images_dir}")
    print(f"Augmentation multiplier: {args.multiplier}x")
    
    # 1. Define Albumentations transformation pipeline
    transform = A.Compose(
        [
            # Random rotation within -30 to 30 degrees
            A.Rotate(limit=30, p=0.5),
            # Random brightness and contrast adjustments
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # Gaussian noise injection
            A.GaussNoise(std_range=(0.01, 0.05), p=0.4),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2  # Keep box only if at least 20% of its area remains
        )
    )

    # 2. Collect original images
    image_paths = sorted([
        p for p in train_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        and "_aug_" not in p.name  # Skip already augmented images if re-run
    ])
    
    print(f"Found {len(image_paths)} original training images.")
    if not image_paths:
        print("No training images found to augment.")
        return

    augmented_count = 0
    
    # 3. Augmentation loop
    for idx, img_path in enumerate(image_paths):
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Find matching label file
        label_name = f"{img_path.stem}.txt"
        label_path = train_labels_dir / label_name
        
        # Load labels
        bboxes, class_labels = load_yolo_labels(label_path)
        
        for i in range(args.multiplier):
            try:
                # Apply transformations
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                aug_image = transformed["image"]
                aug_bboxes = transformed["bboxes"]
                aug_class_labels = transformed["class_labels"]
                
                # Output paths
                aug_img_name = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                aug_img_path = train_images_dir / aug_img_name
                aug_lbl_path = train_labels_dir / f"{img_path.stem}_aug_{i}.txt"
                
                # Write image
                cv2.imwrite(str(aug_img_path), aug_image)
                
                # Write labels
                write_yolo_labels(aug_lbl_path, aug_bboxes, aug_class_labels)
                augmented_count += 1
                
            except Exception as e:
                print(f"Failed to augment {img_path.name} (copy {i}): {e}")

    print("\n" + "="*50)
    print("DATA AUGMENTATION STATISTICS")
    print("="*50)
    print(f"Original images processed: {len(image_paths)}")
    print(f"Augmented images generated: {augmented_count}")
    print(f"Total training images now: {len(image_paths) + augmented_count}")
    print("="*50)
    print("Data augmentation complete.")


if __name__ == "__main__":
    main()
