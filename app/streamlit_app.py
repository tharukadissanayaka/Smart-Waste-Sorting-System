# app/streamlit_app.py
"""
Smart Waste Sorting System — Streamlit Web Application
Course  : EC6301 Artificial Intelligence
Group   : G28 | University of Ruhuna
Author  : Jayasekara T.H.D.P.U. (EG/2022/5098)

Usage:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
import time

# ── Add project root to path so we can import from scripts/ ──────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ── Local utils ───────────────────────────────────────────────────────────────
from app.utils import (
    load_model,
    run_inference,
    CLASS_NAMES,
    CLASS_COLORS,
    assess_contamination_risk,
    FINETUNED_MODEL_PATH,
    FALLBACK_MODEL_PATH,
)

# ── Try to import real contamination logic (Task C) ──────────────────────────
# TODO: Uncomment once scripts/contamination_logic.py is finalized by Task C member
# try:
#     from scripts.contamination_logic import assess_contamination_risk
#     CONTAMINATION_SOURCE = "scripts/contamination_logic.py (Task C)"
# except ImportError:
#     from app.utils import assess_contamination_risk
#     CONTAMINATION_SOURCE = "app/utils.py (stub — Task C not yet integrated)"
CONTAMINATION_SOURCE = "app/utils.py (stub — wire in Task C when ready)"




#related to step 10 when task C is done, we can remove the stub and use the real contamination logic from scripts/contamination_logic.py
try:
    from scripts.contamination_logic import assess_contamination_risk
    st.sidebar.success("✅ Contamination logic: scripts/contamination_logic.py")
except ImportError:
    from app.utils import assess_contamination_risk
    st.sidebar.warning("⚠️ Using contamination stub")



# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Smart Waste Sorting System",
    page_icon  = "♻️",
    layout     = "wide",
)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("app/logo.svg", width=80)
    st.title("⚙️ Settings & Info")

    st.markdown("---")

    # ── Confidence Threshold ─────────────────────────────────────────────────
    st.subheader("🎚️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value = 0.05,
        max_value = 1.00,
        value     = 0.25,
        step      = 0.05,
        help      = "Only show detections above this confidence score.",
    )

    st.markdown("---")

    # ── Model Info ───────────────────────────────────────────────────────────
    st.subheader("🤖 Model Settings")
    model_options = []
    if os.path.exists(FINETUNED_MODEL_PATH):
        model_options.append("Fine-tuned Waste Model")
    model_options.append("Pretrained COCO Model")

    model_choice = st.selectbox(
        "Select Model Checkpoint:",
        options=model_options,
        index=0,
        help="Switch between the fine-tuned waste classification model and the general pretrained COCO model."
    )

    if model_choice == "Fine-tuned Waste Model":
        st.success(f"✅ Active:\n`{FINETUNED_MODEL_PATH}`")
    else:
        st.info(f"ℹ️ Active:\n`{FALLBACK_MODEL_PATH}` (COCO classes)")

    st.markdown(f"**Contamination Logic:** `{CONTAMINATION_SOURCE}`")
    st.markdown("**Classes:** Plastic · Paper · Glass · Metal")
    st.markdown("**Input Size:** 640 × 640 px")

    st.markdown("---")

    # ── Class Color Legend ───────────────────────────────────────────────────
    st.subheader("🎨 Class Color Legend")
    color_names_hex = {
        "Plastic" : "#FFA500",
        "Paper"   : "#00FF00",
        "Glass"   : "#0000FF",
        "Metal"   : "#FF0000",
    }
    for cls, hex_color in color_names_hex.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:18px;height:18px;background:{hex_color};'
            f'border-radius:3px;"></div><span>{cls}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── About ────────────────────────────────────────────────────────────────
    st.subheader("ℹ️ About This Project")
    st.markdown(
        """
        **Smart Waste Sorting System**  
        *EC6301 — Artificial Intelligence*  
        *University of Ruhuna, Group G28*

        This system uses a fine-tuned **YOLOv8** model for 
        automated, real-time waste segregation into four categories: 
        Plastic, Paper, Glass, and Metal.

        It also applies a **contamination logic layer** to flag 
        uncertain or potentially mis-classified items, addressing 
        the challenge of visually similar materials (e.g., 
        clear plastic vs. glass).

        **Target:** mAP@0.5 ≥ 85% | ≥ 15 FPS on CPU
        """
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═════════════════════════════════════════════════════════════════════════════
st.title("♻️ Smart Waste Sorting System")
st.markdown(
    "Upload a waste image to classify it into **Plastic, Paper, Glass, or Metal** "
    "using YOLOv8 object detection."
)

# ── Load Model ────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, model_path_used, is_finetuned = load_model(model_choice)

st.markdown("---")

# ── Input Mode Selection ──────────────────────────────────────────────────────
input_mode = st.radio(
    "Select Input Mode:",
    ["📁 Upload Image", "🎥 Webcam (Live)"],
    horizontal = True,
)

# ═════════════════════════════════════════════════════════════════════════════
# MODE 1: IMAGE UPLOAD
# ═════════════════════════════════════════════════════════════════════════════
if input_mode == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a waste image (JPG or PNG)",
        type = ["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image_pil, width="stretch")

        # ── Run Inference ─────────────────────────────────────────────────
        with st.spinner("Running inference..."):
            detections, fps, annotated_img = run_inference(
                model, image_pil, conf_threshold
            )

        with col2:
            st.subheader("🔍 Detection Results")
            st.image(annotated_img, width="stretch")

        # ── FPS Readout ───────────────────────────────────────────────────
        st.markdown("---")
        fps_col, det_col = st.columns([1, 3])
        with fps_col:
            fps_color = "green" if fps >= 15 else "red"
            st.metric(
                label = "⚡ Inference Speed",
                value = f"{fps:.1f} FPS",
                delta = "✅ Real-time" if fps >= 15 else "⚠️ Below 15 FPS target",
            )
            st.caption(f"Inference time: {1000/fps:.1f} ms" if fps > 0 else "")

        with det_col:
            st.metric("📦 Total Detections", len(detections))

        # ── Results Table ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Detection Details & Contamination Risk")

        if len(detections) == 0:
            st.info(
                "No objects detected. Try lowering the confidence threshold in the sidebar."
            )
        else:
            # Build table data with contamination risk
            table_data = []
            for i, det in enumerate(detections):
                risk = assess_contamination_risk(det["class_name"], det["confidence"])
                table_data.append({
                    "#"                  : i + 1,
                    "Class"              : det["class_name"],
                    "Confidence"         : f"{det['confidence']:.3f}",
                    "Risk Level"         : risk["risk_level"],
                    "Contamination Flag" : "🚩 Yes" if risk["flag"] else "✅ No",
                    "Message"            : risk["message"],
                })

            # Display each row
            for row in table_data:
                with st.expander(
                    f"Detection #{row['#']}: **{row['Class']}** "
                    f"(conf: {row['Confidence']}) — Risk: {row['Risk Level']}"
                ):
                    st.markdown(row["Message"])
                    bbox_idx = row["#"] - 1
                    bbox = detections[bbox_idx]["bbox"]
                    st.caption(
                        f"Bounding Box: x1={bbox[0]:.0f}, y1={bbox[1]:.0f}, "
                        f"x2={bbox[2]:.0f}, y2={bbox[3]:.0f}"
                    )

            # Summary table (compact)
            st.markdown("**Summary Table:**")
            import pandas as pd
            df = pd.DataFrame(table_data).drop(columns=["Message"])
            st.dataframe(df, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# MODE 2: WEBCAM LIVE
# ═════════════════════════════════════════════════════════════════════════════
elif input_mode == "🎥 Webcam (Live)":
    st.info(
        "📸 Click **'Capture Photo'** to take a snapshot from your webcam and run detection."
    )

    webcam_image = st.camera_input("Take a photo")

    if webcam_image is not None:
        image_pil = Image.open(webcam_image).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Captured Frame")
            st.image(image_pil, width="stretch")

        with st.spinner("Running inference on webcam frame..."):
            detections, fps, annotated_img = run_inference(
                model, image_pil, conf_threshold
            )

        with col2:
            st.subheader("🔍 Detection Results")
            st.image(annotated_img, width="stretch")

        st.metric("⚡ FPS (this frame)", f"{fps:.1f}")
        st.metric("📦 Detections Found", len(detections))

        if detections:
            st.subheader("📊 Contamination Risk Assessment")
            for i, det in enumerate(detections):
                risk = assess_contamination_risk(det["class_name"], det["confidence"])
                st.markdown(
                    f"**{i+1}. {det['class_name']}** (conf: {det['confidence']:.3f}) "
                    f"→ {risk['message']}"
                )
        else:
            st.info("No objects detected in this frame.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Smart Waste Sorting System · EC6301 AI · University of Ruhuna · Group G28 · 2026"
)