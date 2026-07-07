# app/utils.py
"""
Utility functions for the Smart Waste Sorting System Streamlit app.
Handles: model loading, bounding box drawing, color mapping per class.
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import time

# ─────────────────────────────────────────────
# CONFIG — swap MODEL_PATH once real checkpoint is ready
# ─────────────────────────────────────────────
FINETUNED_MODEL_PATH = "models/waste_yolov8_best.pt"
FALLBACK_MODEL_PATH  = "yolov8n.pt"   # downloaded automatically by ultralytics

# 4 waste classes as defined in the project
CLASS_NAMES = ["Plastic", "Paper", "Glass", "Metal"]

# Color per class: BGR format for OpenCV  (also used in Streamlit display)
CLASS_COLORS = {
    "Plastic" : (0,   165, 255),   # Orange
    "Paper"   : (0,   255,   0),   # Green
    "Glass"   : (255,   0,   0),   # Blue
    "Metal"   : (0,     0, 255),   # Red
}

# ─────────────────────────────────────────────
# Model Loading (cached so it loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_choice: str = "Fine-tuned Waste Model"):
    """
    Load YOLOv8 model with caching.
    Tries fine-tuned checkpoint first; falls back to base yolov8n.pt.
    Returns: (model, model_path_used, is_finetuned)
    """
    if model_choice == "Fine-tuned Waste Model" and os.path.exists(FINETUNED_MODEL_PATH):
        model_path   = FINETUNED_MODEL_PATH
        is_finetuned = True
    else:
        model_path   = FALLBACK_MODEL_PATH
        is_finetuned = False

    model = YOLO(model_path)
    # Warm up the model with a dummy inference to prevent first-run latency lag
    try:
        model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    except Exception:
        pass
    return model, model_path, is_finetuned


# ─────────────────────────────────────────────
# Color Mapping Helper
# ─────────────────────────────────────────────
def get_class_color(class_name: str) -> tuple:
    """Return BGR color for a given class name."""
    return CLASS_COLORS.get(class_name, (128, 128, 128))  # gray for unknown


# ─────────────────────────────────────────────
# Bounding Box Drawing
# ─────────────────────────────────────────────
def draw_bounding_boxes(image_np: np.ndarray, detections: list, conf_threshold: float = 0.25) -> np.ndarray:
    """
    Draw color-coded bounding boxes on image.

    Args:
        image_np      : NumPy array (RGB)
        detections    : List of dicts with keys:
                        'class_name', 'confidence', 'bbox' (x1,y1,x2,y2)
        conf_threshold: Minimum confidence to display

    Returns:
        Annotated image as NumPy array (RGB)
    """
    img = image_np.copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for det in detections:
        if det["confidence"] < conf_threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        class_name      = det["class_name"]
        confidence      = det["confidence"]
        color           = get_class_color(class_name)   # BGR

        # Draw rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        # Label background
        label     = f"{class_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)

        # Label text (white)
        cv2.putText(img_bgr, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# Run Inference & Measure FPS
# ─────────────────────────────────────────────
def run_inference(model, image_pil: Image.Image, conf_threshold: float = 0.25):
    """
    Run YOLOv8 inference on a PIL image.

    Returns:
        detections  : list of detection dicts
        fps         : frames-per-second for this single inference
        result_img  : annotated NumPy array (RGB)
    """
    image_np = np.array(image_pil.convert("RGB"))

    start_time = time.time()
    results    = model(image_np, conf=conf_threshold, verbose=False)
    elapsed    = time.time() - start_time
    fps        = 1.0 / elapsed if elapsed > 0 else 0.0

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf            = float(box.conf[0])
            cls_id          = int(box.cls[0])

            # Use model's own names (handles both fine-tuned & fallback)
            class_name = model.names.get(cls_id, f"Class_{cls_id}")

            detections.append({
                "class_name" : class_name,
                "confidence" : conf,
                "bbox"       : (x1, y1, x2, y2),
                "class_id"   : cls_id,
            })

    annotated_img = draw_bounding_boxes(image_np, detections, conf_threshold)
    return detections, fps, annotated_img






# ─────────────────────────────────────────────
# Contamination Logic — Stub / Integration
# ─────────────────────────────────────────────
# TODO: Once scripts/contamination_logic.py is ready (Task C - Dissanayake S.S.S.),
#       replace this stub with:
#           import sys, os
#           sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
#           from contamination_logic import assess_contamination_risk
#
# For now, a simple local stub is used so the app runs end-to-end.

def assess_contamination_risk(class_name: str, confidence: float) -> dict:
    """
    STUB: Assess contamination risk for a detected item.
    
    Replace with import from scripts/contamination_logic.py when available.

    Returns dict with:
        'risk_level'  : 'LOW', 'MEDIUM', 'HIGH'
        'flag'        : bool
        'message'     : str
    """
    CONFUSION_PAIRS = {
        ("Plastic", "Glass") : 0.5,
        ("Glass",  "Plastic"): 0.5,
        ("Paper",  "Plastic"): 0.6,
    }

    HIGH_RISK_THRESHOLD   = 0.4
    MEDIUM_RISK_THRESHOLD = 0.6

    # High risk: very low confidence
    if confidence < HIGH_RISK_THRESHOLD:
        return {
            "risk_level" : "HIGH",
            "flag"       : True,
            "message"    : (
                f"⚠️ HIGH RISK: '{class_name}' detected with low confidence "
                f"({confidence:.2f}). Manual inspection recommended."
            ),
        }
    # Medium risk: moderate confidence for confusion-prone classes
    elif confidence < MEDIUM_RISK_THRESHOLD and class_name in ["Plastic", "Glass"]:
        return {
            "risk_level" : "MEDIUM",
            "flag"       : True,
            "message"    : (
                f"🔶 MEDIUM RISK: '{class_name}' (conf {confidence:.2f}) — "
                "possibly confused with visually similar material (e.g., Plastic vs Glass)."
            ),
        }
    else:
        return {
            "risk_level" : "LOW",
            "flag"       : False,
            "message"    : f"✅ LOW RISK: '{class_name}' classified confidently ({confidence:.2f}).",
        }