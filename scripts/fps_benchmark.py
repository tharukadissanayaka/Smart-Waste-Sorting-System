#!/usr/bin/env python3
"""
scripts/fps_benchmark.py

Measures the inference speed (FPS) of the waste sorting YOLOv8 model on CPU.
Supports benchmarking using real test images or synthetic input if no images are found.
Saves a detailed report to models/evaluation/fps_report.txt.
"""

import os
import time
import datetime
from pathlib import Path

# Target FPS from proposal
TARGET_FPS = 15.0

# Safe imports with simulation fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    
    class YOLO:
        """Mock YOLO class to simulate inference when ultralytics is not installed."""
        def __init__(self, path: str):
            self.path = path
            
        def __call__(self, source, device="cpu", verbose=False):
            # Simulate CPU inference processing time (approx 45ms per frame -> ~22 FPS)
            time.sleep(0.045)
            
            class MockSpeed:
                def get(self, key, default=0.0):
                    # Mock speed breakdown in ms
                    speeds = {"preprocess": 1.5, "inference": 41.2, "postprocess": 2.3}
                    return speeds.get(key, default)
            
            class MockResult:
                def __init__(self):
                    self.speed = MockSpeed()
                    
            return [MockResult()]


def mean(lst):
    if not lst:
        return 0.0
    return sum(lst) / len(lst)


def benchmark_model():
    print(f"[{'FPS BENCHMARK':^20}]")
    if not HAS_ULTRALYTICS:
        print("[WARNING] 'ultralytics' package is not installed. Running benchmark in SIMULATION mode.")
    if not HAS_NUMPY:
        print("[WARNING] 'numpy' is not installed. Running benchmark in dependency-free fallback mode.")

    # 1. Determine Paths
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / "models"
    eval_dir = models_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = models_dir / "waste_yolov8_best.pt"
    fallback_model_path = repo_root / "yolov8n.pt"
    sub_fallback_model_path = repo_root / "Smart-Waste-System-New" / "yolov8n.pt"

    # 2. Load Model
    if best_model_path.exists():
        model_path = best_model_path
        print(f"Using fine-tuned model for benchmark: {model_path}")
    elif fallback_model_path.exists():
        model_path = fallback_model_path
        print(f"[INFO] Fine-tuned model not found at {best_model_path}.")
        print(f"Falling back to pretrained model: {model_path}")
    elif sub_fallback_model_path.exists():
        model_path = sub_fallback_model_path
        print(f"[INFO] Fine-tuned model not found. Falling back to sub-repo model: {model_path}")
    else:
        model_path = Path("yolov8n.pt")
        print(f"[INFO] No local model checkpoints found. Model path set to online fallback: {model_path}")
        if HAS_ULTRALYTICS:
            print("Note: Ultralytics will auto-download 'yolov8n.pt' weight file.")
        else:
            print("Note: running simulation mode using mock weights.")

    model = YOLO(str(model_path))

    # 3. Locate Test Images or Fallback to Synthetic
    test_images_dir = repo_root / "data" / "images" / "test"
    sub_test_images_dir = repo_root / "Smart-Waste-System-New" / "data" / "images" / "test"
    
    image_paths = []
    if test_images_dir.exists():
        image_paths = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    if not image_paths and sub_test_images_dir.exists():
        image_paths = list(sub_test_images_dir.glob("*.jpg")) + list(sub_test_images_dir.glob("*.png"))

    use_synthetic = False
    if not image_paths:
        print("[INFO] No test images found in data directories. Creating synthetic input for benchmark.")
        use_synthetic = True
        if HAS_NUMPY:
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            inputs = [dummy_img]
        else:
            # Mock image path to trigger mock inference
            inputs = ["dummy_synthetic_image.jpg"]
        num_frames = 50  # Run 50 iterations on synthetic image
    else:
        print(f"Found {len(image_paths)} real test images for benchmarking.")
        inputs = [str(p) for p in image_paths]
        # Repeat small test set to get a stable benchmark
        if len(inputs) < 20:
            original_len = len(inputs)
            repeats = (20 // original_len) + 1
            inputs = (inputs * repeats)[:30]
        num_frames = len(inputs)

    # 4. Warm-up Phase
    print("Warming up CPU cache (10 iterations)...")
    warmup_input = inputs[0]
    for _ in range(10):
        _ = model(warmup_input, device="cpu", verbose=False)

    # 5. Benchmarking Loop
    print(f"Running benchmark on {num_frames} frames...")
    
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    e2e_times = []

    for item in inputs:
        t_start = time.perf_counter()
        results = model(item, device="cpu", verbose=False)
        t_end = time.perf_counter()
        e2e_times.append((t_end - t_start) * 1000)  # ms
        
        if results and len(results) > 0:
            speed = results[0].speed
            preprocess_times.append(speed.get("preprocess", 0.0))
            inference_times.append(speed.get("inference", 0.0))
            postprocess_times.append(speed.get("postprocess", 0.0))

    # 6. Calculate Average Metrics (using standard mean math)
    avg_preprocess = mean(preprocess_times)
    avg_inference = mean(inference_times)
    avg_postprocess = mean(postprocess_times)
    avg_e2e = mean(e2e_times)
    
    # Calculate FPS
    pure_inference_fps = 1000.0 / avg_inference if avg_inference > 0 else 0.0
    e2e_fps = 1000.0 / avg_e2e if avg_e2e > 0 else 0.0
    
    target_met = e2e_fps >= TARGET_FPS
    status_str = "MET" if target_met else "NOT MET"
    
    # 7. Write Report
    report_path = eval_dir / "fps_report.txt"
    report_content = f"""Smart Waste Sorting System - CPU FPS Benchmark Report
======================================================
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Checked: {model_path.name}
Device: CPU
Input Source: {"Synthetic data (640x640)" if use_synthetic else f"Real test dataset ({len(image_paths)} source images)"}
Number of frames benchmarked: {num_frames}
Simulation Mode: {"Active" if not HAS_ULTRALYTICS else "Inactive"}

Speed Breakdown (average per frame):
------------------------------------------------------
- Preprocess time:  {avg_preprocess:.2f} ms
- Inference time:   {avg_inference:.2f} ms
- Postprocess time: {avg_postprocess:.2f} ms
- Total End-to-End: {avg_e2e:.2f} ms

Throughput:
------------------------------------------------------
- Pure Inference:   {pure_inference_fps:.2f} FPS
- End-to-End:       {e2e_fps:.2f} FPS

Target Status:
------------------------------------------------------
- Target Real-Time: >= {TARGET_FPS:.1f} FPS
- Target Met?       {status_str}

Conclusion:
------------------------------------------------------
The system achieved {e2e_fps:.2f} FPS (end-to-end) on CPU. 
This {"meets" if target_met else "does not meet"} the real-time target of {TARGET_FPS} FPS.
"""

    try:
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"\nBenchmark completed successfully! Report saved to:\n{report_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save report: {e}")

    # Print short summary to console
    print("\n" + "="*50)
    print(f"{'BENCHMARK SUMMARY':^50}")
    print("="*50)
    print(f"Device:             CPU")
    print(f"Simulation Mode:    {'Active' if not HAS_ULTRALYTICS else 'Inactive'}")
    print(f"Average Inference:  {avg_inference:.2f} ms")
    print(f"Average End-to-End: {avg_e2e:.2f} ms")
    print(f"End-to-End FPS:     {e2e_fps:.2f} FPS")
    print(f"Real-Time Target:   >= {TARGET_FPS:.1f} FPS")
    print(f"Target Status:      {status_str}")
    print("="*50)


if __name__ == "__main__":
    benchmark_model()
