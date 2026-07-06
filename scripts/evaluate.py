#!/usr/bin/env python3
"""
Script to evaluate the trained YOLOv8 model on the test split.

Outputs:
- Computes mAP, Precision, Recall, IoU.
- Generates Confusion Matrix and PR curves using Ultralytics built-in evaluation.
- Prints a summary table to the console.
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Default paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "data.yaml"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
BEST_MODEL_PATH = DEFAULT_MODELS_DIR / "waste_yolov8_best.pt"
FALLBACK_MODEL_PATH = REPO_ROOT / "yolov8n.pt"
EVAL_DIR = DEFAULT_MODELS_DIR / "evaluation"

def main():
    print(f"[{'EVALUATION':^20}]")
    
    # 1. Load Model
    if BEST_MODEL_PATH.exists():
        print(f"Loading trained model: {BEST_MODEL_PATH}")
        model = YOLO(str(BEST_MODEL_PATH))
    else:
        print(f"[WARNING] Trained model not found at {BEST_MODEL_PATH}.")
        print(f"Falling back to pretrained model: {FALLBACK_MODEL_PATH}")
        print("Note: Metrics will be meaningless until the fine-tuned checkpoint is dropped in.")
        model = YOLO(str(FALLBACK_MODEL_PATH))

    if not DEFAULT_DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset config not found at {DEFAULT_DATA_YAML}")

    # 2. Run Evaluation
    print(f"Running evaluation on test split using {DEFAULT_DATA_YAML}...")
    
    # Ultralytics val() automatically generates confusion matrix and PR curves.
    # We specify split='test' to evaluate on test images.
    results = model.val(
        data=str(DEFAULT_DATA_YAML),
        split="test",
        project=str(DEFAULT_MODELS_DIR),
        name="evaluation",
        exist_ok=True, # Overwrite / reuse the same folder
        device="cpu"
    )

    # 3. Print Clean Summary Table
    print("\n" + "="*60)
    print(f"{'EVALUATION SUMMARY (TEST SPLIT)':^60}")
    print("="*60)
    
    # Check if results contains the metrics we need
    box = results.box
    if box is not None:
        # Global metrics
        print(f"Global mAP@0.5      : {box.map50:.4f}")
        print(f"Global mAP@0.5:0.95 : {box.map:.4f}")
        print(f"Global Precision    : {box.mp:.4f}")
        print(f"Global Recall       : {box.mr:.4f}")
        
        print("\n" + "-"*60)
        print(f"{'Class Name':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10}")
        print("-" * 60)
        
        class_names = results.names
        # box.ap_class_index gives the classes that were evaluated
        for i, c in enumerate(box.ap_class_index):
            name = class_names[c]
            p = box.p[i]
            r = box.r[i]
            map50 = box.ap50[i]
            print(f"{name:<20} | {p:<10.4f} | {r:<10.4f} | {map50:<10.4f}")
    else:
        print("No evaluation metrics returned. Check if the test set has labels.")

    print("="*60)
    print(f"Evaluation artifacts (Confusion Matrix, PR Curves) saved to:\n{EVAL_DIR}")
    
    cm_path = EVAL_DIR / "confusion_matrix.png"
    if not cm_path.exists():
        print(f"[WARNING] Expected confusion matrix at {cm_path} but it was not found.")

if __name__ == "__main__":
    main()
