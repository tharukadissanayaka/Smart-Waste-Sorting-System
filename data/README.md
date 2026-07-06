# Smart Waste Sorting System - Data Pipeline

This folder contains the structured dataset and the data processing pipeline for the YOLOv8-based Smart Waste Sorting System.

## Dataset Sources

Our data pipeline integrates two primary datasets:

1. **Garbage Classification v2**
   - **Source:** [Kaggle sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
   - **Description:** ~2500 household waste classification images divided into categories such as Glass, Paper, Cardboard, Plastic, Metal, and Trash.
   - **Citation:**
     ```
     @misc{garbage-classification-v2,
       author = {Sumn2u},
       title = {Garbage Classification V2},
       year = {2020},
       publisher = {Kaggle},
       journal = {Kaggle Dataset},
       howpublished = {\url{https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2}}
     }
     ```

2. **TACO (Trash Annotations in Context)**
   - **Source:** [pedropro/TACO](https://github.com/pedropro/TACO)
   - **Description:** A growing, open-image dataset of waste in the wild. It contains high-resolution images of litter in diverse environments (streets, parks, beaches) with fine-grained bounding boxes and segmentation masks.
   - **Citation:**
     ```
     @article{proencca2020taco,
       title={TACO: Trash Annotations in Context for Litter Detection},
       author={Proen{\c{c}}a, Pedro F and Sim{\~o}es, Francisco},
       journal={arXiv preprint arXiv:2003.01290},
       year={2020}
     }
     ```

---

## Class Mapping Decisions

To align with the project proposal's methodology, the fine-grained classes from both source datasets are mapped down to **four target categories**:

| Target Class ID | Target Class Name | Source Mappings (Kaggle & TACO) | Description & Rationale |
| :--- | :--- | :--- | :--- |
| `0` | **Plastic** | `Plastic`, `clear plastic bottle`, `other plastic bottle`, `plastic bag`, `plastic film`, `plastic wrapper`, `plastic straw`, etc. | All plastic containers, bags, and packaging materials. |
| `1` | **Paper** | `Paper`, `Cardboard`, `cardboard`, `paper bag`, `carton`, `paper cup`, etc. | Combined recyclable paper products. Cardboard is merged here. |
| `2` | **Glass** | `Glass`, `glass bottle`, `glass jar`, `broken glass`, etc. | Glass bottles, jars, and other glass items. |
| `3` | **Metal** | `Metal`, `soda can`, `aluminium foil`, `pop tab`, `battery`, `organic`, `trash`, `biological`, `cigarette`, etc. | Acts as the **Metal/Organic** bucket. All metals, organic materials, non-recyclables, and general litter are merged here as per the proposal's methodology. |

---

## Dataset Statistics

*Note: The following statistics reflect the output of running the full/sample pipeline.*

### Image Counts per Split & Class

| Split | Total Images | Plastic (0) | Paper (1) | Glass (2) | Metal/Organic (3) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 114 | 15 | 24 | 18 | 57 |
| **Validation** | 11 | 2 | 2 | 0 | 7 |
| **Test** | 6 | 0 | 0 | 0 | 6 |
| **Total** | **131** | **17** | **26** | **18** | **70** |

*(These stats include 76 augmented images added to the **train** split using Albumentations).*

---

## How to Re-Run the Pipeline

Follow these steps to download the raw datasets, prepare the YOLO format splits, and apply the training data augmentations.

### Prerequisites

Ensure you have installed the required python packages:
```bash
pip install -r requirements.txt
```

#### Setting up Kaggle API Credentials (Optional)
To download the real Kaggle dataset, you need a Kaggle API key:
1. Navigate to your Account Settings on Kaggle and click **Create New API Token** to download `kaggle.json`.
2. Save this file to `~/.kaggle/kaggle.json` (on Windows: `C:\Users\<Username>\.kaggle\kaggle.json`).
*If credentials are not found, the downloader script will fall back to generating synthetic/dummy data for testing purposes.*

### Step 1: Download / Generate Data
To download the real datasets (limit TACO downloads to 50 images for speed):
```bash
python scripts/download_data.py --taco-limit 50
```
Or run in **dry-run mode** using synthetic data (does not require Kaggle credentials or network access):
```bash
python scripts/download_data.py --dry-run
```

### Step 2: Prepare YOLO Dataset Format
Convert the raw dataset downloads to YOLO format and perform train/val/test splits (70/20/10):
```bash
python scripts/prepare_yolo_dataset.py
```
This generates:
- `data/images/train/`, `data/images/val/`, `data/images/test/`
- `data/labels/train/`, `data/labels/val/`, `data/labels/test/`
- `data/data.yaml` (Ultralytics dataset configuration)

### Step 3: Augment Training Data
Apply Albumentations (rotation, brightness/contrast jitter, Gaussian noise) to the **train** split images:
```bash
python scripts/augment_data.py --multiplier 2
```
This will generate 2 augmented variations for every original image in the training split.
