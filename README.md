# Brain Tumor Detection System 🧠🔬

An end-to-end Deep Learning & Computer Vision application for automated classification of Brain Tumor MRI scans. Built using **TensorFlow/Keras**, **OpenCV**, and **Flask**.

---

## 🌟 Key Features

- **Multi-Model Deep Learning Pipeline**: Evaluates and compares multiple neural network architectures:
  - **VGG16 Transfer Learning** (Best Performing: ~84.3% Accuracy)
  - **MobileNetV2** (Lightweight Architecture: ~80.4% Accuracy)
  - **ResNet50** (Residual Network: ~78.4% Accuracy)
  - **Custom Convolutional Neural Network (CNN)** (~66.7% Accuracy)
- **Automated Image Preprocessing**: Normalization, resizing to 224x224, and color space transformations via OpenCV.
- **Interactive Flask Web Application**: User-friendly web interface allowing real-time MRI upload, diagnosis, confidence score display, and model selection.
- **Model Evaluation & Performance Metrics**: Detailed visualizations including Confusion Matrices, ROC Curves, and Training History charts.

---

## 📊 Model Comparison & Performance

| Model Architecture | Type | Performance Accuracy | Description |
| :--- | :--- | :--- | :--- |
| **VGG16 Transfer Learning** | Transfer Learning | **84.31%** | Deep CNN pre-trained on ImageNet fine-tuned for MRI scans |
| **MobileNetV2** | Transfer Learning | **80.39%** | Efficient lightweight mobile architecture |
| **ResNet50** | Transfer Learning | **78.43%** | Deep residual network architecture |
| **Custom CNN** | Custom Deep Architecture | **66.67%** | Multi-layer custom convolutional network |

---

## 📁 Repository Structure

```
├── app.py                      # Core Flask Application & API Endpoints
├── requirements.txt            # Python Dependencies
├── commands.txt                # Quick launch instructions
├── Brain_Tumor_Detection_Complete.ipynb # Complete Model Training Notebook
├── models/
│   └── model_comparison.csv    # Evaluated Model Accuracy Metrics
├── templates/
│   ├── base.html              # Base layout template
│   ├── index.html             # Homepage & upload portal
│   ├── predict.html           # Prediction result view
│   ├── visualization.html     # Performance metrics & comparison charts
│   └── about.html             # Project details
├── static/
│   ├── css/style.css          # Application UI Styles
│   ├── js/main.js             # Client-side interactions
│   └── uploads/               # Temporary MRI uploads
├── confusion_matrices.png      # Evaluated confusion matrices
├── model_comparison_bars.png   # Model accuracy bar chart
├── roc_curves.png              # Receiver Operating Characteristic curves
└── training_history_loss.png   # Training vs Validation loss history
```

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.10 to 3.13 is installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harhs4344/Brain-Tumor-Detection-System.git
   cd Brain-Tumor-Detection-System
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:5000` (or `http://localhost:8080`).

---

## 💻 Web Application Usage

1. Open the homepage at `http://localhost:5000`.
2. Click **Upload MRI Scan** and select a PNG/JPG/JPEG Brain MRI image.
3. Select your preferred Deep Learning model (e.g. VGG16 Transfer Learning).
4. View the diagnostic prediction, confidence probability percentage, and model evaluation summary.

---

## 📜 License & Citation

Developed for AI & Data Science research and educational evaluation.
