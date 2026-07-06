#!/usr/bin/env python3
"""
Script to train / fine-tune a YOLOv8 object detection model on the waste sorting dataset.
Uses transfer learning from a pretrained Ultralytics checkpoint.

Ultralytics default loss functions:
- Complete IoU (CIoU) loss for bounding box regression.
- Binary Cross Entropy (BCE) loss for category classification.
- Distribution Focal Loss (DFL) for box boundary regression representation.
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

# Default relative paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "data.yaml"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on the smart waste sorting dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Path to data.yaml dataset config file."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default 640)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size: n (nano), s (small), m (medium), l (large), x (xlarge)."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory to save training run logs and checkpoints."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="training_results",
        help="Subdirectory name under --project for this run's outputs."
    )
    return parser


def main():
    parser = setup_args()
    args = parser.parse_args()
    
    # Resolve paths
    data_path = args.data.resolve()
    project_dir = args.project.resolve()
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at {data_path}. "
            "Please make sure you have run the dataset preparation script first."
        )

    # 1. Load pretrained model
    model_name = f"yolov8{args.model_size}.pt"
    print(f"Loading pretrained model: {model_name}...")
    model = YOLO(model_name)

    # 2. Run transfer learning
    print(f"Starting training on {data_path}...")
    print(f"Hyperparameters: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, lr0={args.lr}")
    
    # Run training
    # CIoU loss and BCE loss are used by default inside Ultralytics YOLOv8 detection head.
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        project=str(project_dir),
        name=args.name,
        device="cpu",  # default to CPU for portability/dry-runs; can be updated to GPU (device=0)
        workers=0      # 0 workers avoids multiprocessing issues on Windows/laptops
    )

    print("\nTraining completed successfully!")

    # 3. Locate best and last checkpoints
    weights_dir = project_dir / args.name / "weights"
    best_weights_src = weights_dir / "best.pt"
    last_weights_src = weights_dir / "last.pt"

    best_weights_dst = project_dir / "waste_yolov8_best.pt"
    last_weights_dst = project_dir / "waste_yolov8_last.pt"

    # 4. Copy checkpoints directly to models/ root
    if best_weights_src.exists():
        shutil.copy2(best_weights_src, best_weights_dst)
        print(f"Copied best checkpoint to: {best_weights_dst}")
    else:
        print(f"[WARNING] Best weights file not found at {best_weights_src}")

    if last_weights_src.exists():
        shutil.copy2(last_weights_src, last_weights_dst)
        print(f"Copied last checkpoint to: {last_weights_dst}")
    else:
        print(f"[WARNING] Last weights file not found at {last_weights_src}")

    # 5. Print out resulting metrics summary
    print("\n" + "="*50)
    print("TRAINING PERFORMANCE SUMMARY")
    print("="*50)
    if hasattr(results, "results_dict") and results.results_dict:
        for metric, val in results.results_dict.items():
            print(f"  {metric}: {val:.4f}")
    else:
        print("  Metrics summary not found in results object. Check models/training_results/ for files.")
    print("="*50)


if __name__ == "__main__":
    main()
