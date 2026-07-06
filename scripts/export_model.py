#!/usr/bin/env python3
"""
Script to export a trained YOLOv8 PyTorch model (.pt) to ONNX format
for edge/CPU-optimized deployments. Reports file sizes.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO

# Default relative paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "waste_yolov8_best.pt"


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained YOLOv8 model to ONNX format."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to PyTorch model checkpoint (.pt) to export."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="Target format to export (default: onnx)."
    )
    return parser


def main():
    parser = setup_args()
    args = parser.parse_args()
    
    model_path = args.model.resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at {model_path}. "
            "Please run scripts/train.py first before exporting."
        )

    # 1. Load trained PyTorch model
    print(f"Loading trained model from: {model_path}...")
    model = YOLO(str(model_path))

    # 2. Get PyTorch model file size
    pt_size_bytes = os.path.getsize(model_path)
    pt_size_mb = pt_size_bytes / (1024 * 1024)
    print(f"Original PyTorch (.pt) file size: {pt_size_mb:.2f} MB")

    # 3. Export to target format (ONNX)
    print(f"Exporting model to {args.format.upper()} format (this may take a minute)...")
    try:
        # model.export() returns the path to the exported file
        exported_path_str = model.export(format=args.format, dynamic=False)
        exported_path = Path(exported_path_str)
        
        # 4. Get exported file size and report comparison
        if exported_path.exists():
            onnx_size_bytes = os.path.getsize(exported_path)
            onnx_size_mb = onnx_size_bytes / (1024 * 1024)
            
            print("\n" + "="*50)
            print("MODEL EXPORT COMPLETED SUCCESSFULLY")
            print("="*50)
            print(f"  Source Model (.pt) Path:   {model_path}")
            print(f"  Source Model (.pt) Size:   {pt_size_mb:.2f} MB")
            print(f"  Exported Model ({args.format}) Path: {exported_path}")
            print(f"  Exported Model ({args.format}) Size: {onnx_size_mb:.2f} MB")
            print(f"  Size Ratio (Exported/Source):  {onnx_size_mb / pt_size_mb:.2%}")
            print("="*50)
        else:
            print(f"[WARNING] Export completed, but expected file not found at: {exported_path}")
            
    except Exception as e:
        print(f"[ERROR] Model export failed: {e}")


if __name__ == "__main__":
    main()
