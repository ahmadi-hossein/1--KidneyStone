# 🧠 Kidney Stone Detection AI

Welcome to the **Kidney Stone Detection** project – an AI-powered image classification system designed to identify kidney stones in medical images using deep learning.

This project is part of the **IMT ChallengeHub** by [Mohammadreza Momeni](https://github.com/MrezaMomeni) and aims to showcase high-performance medical image classification using PyTorch and custom CNN models.

## Introduction

Kidney stone detection is a critical step in urology and radiology. In this first phase, we focus on building a robust image classification model using state-of-the-art deep learning techniques. The goal is to accurately distinguish kidney stone images from normal X-rays, providing a foundation for precise object localization in the next phase.

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


 Migrate to ResNet18 (Transfer Learning)

 Integrate Grad-CAM for explainability

 Deploy via Streamlit or Gradio

 Evaluate on more real-world samples

### 📂 Dataset

- Source: [Kidney Stone Dataset on Kaggle](https://www.kaggle.com/datasets/mansoordaku/urinary-tract-stone-detection-and-classification)
- Structure: `train/`, `test/`, `labels.csv`
- Format: JPG + YOLO-formatted annotations

### Technologies
## 🔄 Pipeline Overview

1. 📊 **Exploratory Data Analysis (EDA)**
2. 🧪 **Data Augmentation** using Albumentations
3. 🧹 **Data Preprocessing** and Normalization
4. 🧠 **Model Design** (Custom CNN with Keras)
5. 🎯 **Training & Evaluation**
6. 🧾 **Results Visualization** (accuracy, loss, confusion matrix)

All steps are implemented in [this Kaggle notebook](https://www.kaggle.com/code/ahmadihossein/kidney-stone-detection).

### Results
## ✅ Results
- Test Accuracy: **100%**
- Metrics: Precision, Recall, F1-score available in the notebook
- Confusion Matrix: Perfect classification

> 📌 Next step: Move from classification to detection using YOLOv8.

### License / Contribution / Contact
## 📬 Contact
For collaboration or questions, feel free to reach out:
- **Author**: Hossein Ahmadi
- **Email**: Available if any request
- **Kaggle**: [@ahmadihossein](https://www.kaggle.com/ahmadihossein)

---

## 📄 License
This project is licensed under the MIT License.

## 📬 Contact
Developed with ❤️ by Hossein Ahmadi
🌐 Kaggle: ahmadihossein
📧 Email: available on request

🌟 Star this project if you find it useful!
