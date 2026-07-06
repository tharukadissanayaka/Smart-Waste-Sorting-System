#!/usr/bin/env python3
"""
Script to perform a lightweight hyperparameter grid search for YOLOv8.
Tunes batch size and initial learning rate over a small search space.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# Default relative paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "data.yaml"
DEFAULT_TUNING_DIR = REPO_ROOT / "models" / "tuning"


def setup_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a lightweight grid search for YOLOv8 hyperparameters."
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
        default=1,
        help="Number of epochs to train for each grid configuration (default 1 for speed)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=128,
        help="Input image size (default 128 for speed during tuning/smoke tests)."
    )
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=DEFAULT_TUNING_DIR,
        help="Directory to save tuning run results."
    )
    return parser


def main():
    parser = setup_args()
    args = parser.parse_args()
    
    data_path = args.data.resolve()
    tuning_dir = args.tuning_dir.resolve()
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at {data_path}. "
            "Please make sure you have run the dataset preparation script first."
        )

    # Small search space for CPU/laptop constraints
    learning_rates = [0.01, 0.001]
    batch_sizes = [8, 16]
    
    results_grid = []
    best_map50 = -1.0
    best_config = None
    
    print("="*60)
    print("STARTING HYPERPARAMETER GRID SEARCH")
    print(f"Search space: learning_rates={learning_rates}, batch_sizes={batch_sizes}")
    print(f"Config: epochs={args.epochs}, imgsz={args.imgsz}")
    print("="*60)

    for lr in learning_rates:
        for batch in batch_sizes:
            run_name = f"lr_{lr}_batch_{batch}"
            print(f"\n[RUN] Training configuration: Learning Rate = {lr}, Batch Size = {batch}...")
            
            # Initialize a fresh model from pretrained yolov8n.pt for each combination
            model = YOLO("yolov8n.pt")
            
            try:
                # Train the model
                results = model.train(
                    data=str(data_path),
                    epochs=args.epochs,
                    batch=batch,
                    imgsz=args.imgsz,
                    lr0=lr,
                    project=str(tuning_dir),
                    name=run_name,
                    device="cpu",
                    workers=0,
                    plots=False  # Save time and disk by skipping plots during tuning
                )
                
                # Extract validation mAP@0.5 (or box/cls losses)
                map50 = 0.0
                if hasattr(results, "results_dict") and results.results_dict:
                    map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
                    print(f"[SUCCESS] Config {run_name} achieved mAP@0.5 = {map50:.4f}")
                else:
                    print(f"[WARNING] Results dict not found for {run_name}")
                
                results_grid.append({
                    "learning_rate": lr,
                    "batch_size": batch,
                    "mAP50": map50,
                    "status": "success"
                })
                
                if map50 > best_map50:
                    best_map50 = map50
                    best_config = {"learning_rate": lr, "batch_size": batch}
                    
            except Exception as e:
                print(f"[ERROR] Failed configuration {run_name}: {e}")
                results_grid.append({
                    "learning_rate": lr,
                    "batch_size": batch,
                    "mAP50": 0.0,
                    "status": f"failed: {e}"
                })

    # Print final tuning report
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING REPORT")
    print("="*60)
    print(f"{'LR':<10} | {'Batch Size':<10} | {'mAP@0.5':<10} | {'Status':<15}")
    print("-"*60)
    for res in results_grid:
        print(f"{res['learning_rate']:<10} | {res['batch_size']:<10} | {res['mAP50']:<10.4f} | {res['status']:<15}")
    print("-"*60)
    if best_config:
        print(f"Best Configuration Found:")
        print(f"  Learning Rate: {best_config['learning_rate']}")
        print(f"  Batch Size:    {best_config['batch_size']}")
        print(f"  mAP@0.5:       {best_map50:.4f}")
    else:
        print("No successful configurations were completed.")
    print("="*60)


if __name__ == "__main__":
    main()
