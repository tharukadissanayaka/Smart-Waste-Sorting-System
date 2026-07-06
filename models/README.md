# Waste Sorting Model Training Pipeline

This directory holds the fine-tuned model checkpoints (both native PyTorch `.pt` and exported ONNX format) and training results/metrics logs.

## Model Configuration

### Base Model
- **Base Checkpoint:** `yolov8n.pt` (YOLOv8 Nano, ~3.2M parameters)
- **Rationale:** Optimized for CPU, mobile, and resource-constrained edge computing environments (such as a Raspberry Pi or edge camera).

### Training Hyperparameters
- **Epochs:** 50 (default for standard training)
- **Batch Size:** 16
- **Image Size:** 640x640 pixels (normalized and padded automatically)
- **Optimizer:** SGD / AdamW (auto-selected by Ultralytics based on training size)
- **Learning Rate (lr0):** 0.01 (initial learning rate)
- **Loss Functions:**
  - **Complete IoU (CIoU) Loss:** Bounding box coordinate regression.
  - **Binary Cross Entropy (BCE) Loss:** Multi-class classification.
  - **Distribution Focal Loss (DFL):** Continuous distribution boundary regression.

---

## Training Performance

*The following metrics were achieved during the quick smoke test (2 epochs) on the synthetic sample dataset:*

### Loss & Metrics Summary
- **Box Loss (val):** 0.2917
- **Class Loss (val):** 2.9173
- **DFL Loss (val):** 0.1953
- **mAP@0.5 (val):** 0.5492
- **mAP@0.5:0.95 (val):** 0.5492

Training curves (loss and mAP progression plots) are generated and saved under `models/training_results/` (e.g. `results.png`).

---

## Model Exports (ONNX)

For edge deployment, the trained PyTorch checkpoints are converted to the **ONNX (Open Neural Network Exchange)** format.
ONNX models run efficiently on cross-platform inference engines like `onnxruntime`.

- **Source Checkpoint (`.pt`):** `models/waste_yolov8_best.pt`
- **Exported ONNX (`.onnx`):** `models/waste_yolov8_best.onnx`

### File Size Comparison
- **PyTorch (`.pt`) Size:** 5.91 MB
- **ONNX (`.onnx`) Size:** 11.54 MB
- **Ratio (ONNX/PT):** 195.42%

---

## Production / Full Training Requirements

To achieve a production-grade model with **mAP@0.5 >= 85%**, the following compute resources, dataset size, and schedule are recommended:

1. **Hardware:**
   - Dedicated GPU with >= 8GB VRAM (e.g., NVIDIA RTX 3060/4060, or a cloud instance like Google Cloud T4/A100).
   - CPU training is not viable for large-scale training (takes ~20-50x longer).

2. **Dataset Size:**
   - A fully annotated dataset consisting of at least 3,000 to 5,000 real-world, cluttered waste images.
   - Handled via Roboflow or similar manual annotation tools to replace full-image classification fallbacks with tight, item-specific bounding boxes.

3. **Training Configuration:**
   - **Epochs:** 100 to 150 epochs.
   - **Batch Size:** 32 or 64.
   - **Pretrained Checkpoint:** `yolov8s.pt` (small) or `yolov8m.pt` (medium) for a better capacity/speed balance.
   - **Augmentation Multiplier:** 3x or 4x with diverse Albumentations transforms.

---

## How to Reproduce Training

Ensure requirements are installed and data has been preprocessed:
```bash
pip install -r requirements.txt
python scripts/download_data.py --dry-run
python scripts/prepare_yolo_dataset.py
python scripts/augment_data.py
```

### Run Standard Training
To run a standard 50-epoch training on your dataset:
```bash
python scripts/train.py --epochs 50 --batch 16 --imgsz 640 --lr 0.01 --model-size n
```

### Run Hyperparameter Tuning
To sweep parameters over learning rate and batch size:
```bash
python scripts/hyperparameter_tuning.py --epochs 10 --imgsz 320
```

### Run Export to ONNX
Once training finishes, export the best checkpoint to ONNX:
```bash
python scripts/export_model.py
```
