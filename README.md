# Smart Waste Sorting System

An automated, real-time waste segregation and classification system powered by computer vision. This project uses the YOLOv8 object detection model to identify and sort waste into four distinct categories: **Plastic, Paper, Glass, and Metal**. 

By running inference in real-time, the system aims to improve recycling efficiency and reduce contamination at sorting facilities or in smart bins.

---

## 📁 Project Structure

The repository is organized into the following primary directories:

- **`app/`**: Contains the code for the user-facing Streamlit web application. This allows users to upload images or video feeds and see the model's predictions in real-time.
- **`data/`**: Holds the YOLO-formatted dataset (training, validation, and test splits) under `images/` and `labels/`, as well as the main dataset configuration file (`data.yaml`).
- **`models/`**: Stores the trained YOLOv8 PyTorch checkpoints (`.pt`), exported edge-deployment models (`.onnx`), and generated evaluation outputs (like confusion matrices and PR curves).
- **`scripts/`**: Contains all the core logic and pipelines for the project:
  - `train.py`: Fine-tunes the YOLOv8 model via transfer learning.
  - `hyperparameter_tuning.py`: Runs search grids to find optimal training configurations.
  - `export_model.py`: Exports the trained model to ONNX.
  - `evaluate.py`: Evaluates the model on unseen test data and generates performance metrics.

---

## 🚀 Setup & Installation

It is highly recommended to use a virtual environment (like `venv` or `conda`) to manage your dependencies.

**1. Clone the repository & navigate into it:**
```bash
git clone https://github.com/tharukadissanayaka/Smart-Waste-Sorting-System.git
cd Smart-Waste-Sorting-System
```

**2. Create and activate a virtual environment (Optional but recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🛠️ Running the Full Pipeline

The project is designed to be run sequentially from training to evaluation to deployment.

### 1. Training the Model
To fine-tune the YOLOv8 model on the waste dataset, run the training script. This script will automatically save the best and last checkpoints to the `models/` directory.
```bash
python scripts/train.py
```

### 2. Evaluating the Model
Once the model is trained, evaluate its performance on the test dataset. This will output mAP, Precision, and Recall metrics, and save visualization charts (Confusion Matrix, PR Curves) to `models/evaluation/`.
```bash
python scripts/evaluate.py
```

### 3. Running the Web Application
Launch the interactive Streamlit interface to test the model on your own images or webcam feed.
```bash
streamlit run app/streamlit_app.py
```
*(Note: The web app will launch automatically in your default web browser.)*
