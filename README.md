# 🧠 Kidney Stone Detection from Medical Images

Welcome to the **Kidney Stone Detection** project – an AI-powered image classification system designed to identify kidney stones in medical images using deep learning.

This project is part of the **IMT ChallengeHub** by [Mohammadreza Momeni](https://github.com/MrezaMomeni) and aims to showcase high-performance medical image classification using PyTorch and custom CNN models.

---

## 📂 Project Structure
1--KidneyStone/
│
├── notebooks/
│ └── kidney-stone-detection.ipynb # Core notebook
│
├── models/ # Saved trained models
│
├── outputs/ # Visual outputs (confusion matrix, sample predictions)
│
├── data/ # Dataset info or instructions to download
│
├── requirements.txt # Python dependencies
│
└── README.md # This file


---

## 🔍 Problem Statement

Kidney stones are a common urological disorder. Detecting them accurately in medical images (e.g., CT scans, ultrasound) is crucial for timely treatment. This project builds a deep learning model to:

- Classify images as **Stone / No Stone**
- Enable reproducible detection with high accuracy
- Visualize key metrics (confusion matrix, sample predictions)

---

## 🧠 Model Summary

- **Framework**: PyTorch
- **Architecture**: Custom Convolutional Neural Network (CNN)
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss
- **Data Augmentation**: Albumentations (Rotation, Flip, Brightness, etc.)
- **Accuracy**: Achieved 100% test accuracy on internal evaluation

---

## 📊 Results

| Metric            | Value       |
|-------------------|-------------|
| Test Accuracy     | ✅ 100%     |
| Confusion Matrix  | ✅ Included |
| Sample Predictions| ✅ Visualized |

You can find all results in the `outputs/` folder or within the notebook.

---

## 🗃️ Dataset

- ✅ **Source**: [Kaggle - Kidney Stone Detection Dataset][https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection)
- ⚠️ Dataset is not included in this repo due to size and Kaggle's policy.  
- 📥 To run the code, download the dataset manually and place it in the `data/` folder.

---

## 🚀 How to Run

> Python ≥ 3.8 recommended

### 1. Clone the repository

```bash
git clone https://github.com/ahmadi-hossein/1--KidneyStone.git
cd 1--KidneyStone

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download the dataset

### 4. Run the notebook
Use Jupyter or VS Code to open.


