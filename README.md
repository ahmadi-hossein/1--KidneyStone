# 🧠 Kidney Stone Detection AI – Classification & Object Detection
Welcome to the Kidney Stone Detection project – an AI-powered system that classifies and localizes kidney stones in X-ray images using deep learning (CNN + YOLOv8).

This project is part of the **IMT ChallengeHub** by [Mohammadreza Momeni](https://github.com/MrezaMomeni) and aims to showcase high-performance medical image classification using PyTorch and custom CNN models.

!(https://github.com/ahmadi-hossein/1--KidneyStone/blob/main/__results___31_0.png)

## Introduction

Kidney stone detection is a critical step in urology and radiology. In this first phase, we focus on building a robust image classification model using state-of-the-art deep learning techniques. The goal is to accurately distinguish kidney stone images from normal X-rays, providing a foundation for precise object localization in the next phase.

---

## 🔍 Problem Statement

Kidney stones are a common urological disorder. Detecting them accurately in medical images (e.g., CT scans, ultrasound) is crucial for timely treatment. This project builds a deep learning model to:

- Classify images as **Stone / No Stone**
- Enable reproducible detection with high accuracy
- Visualize key metrics (confusion matrix, sample predictions)

---

## 🧠 Model Summary

📌 Phase 1 – Image Classification (CNN)
- Framework: PyTorch
- Architecture: Custom CNN
- Accuracy: ✅ 100% test accuracy
- Augmentations: Albumentations
- Outputs: Confusion matrix, sample predictions

📌 Phase 2 – Object Detection (YOLOv8)
- Framework: Ultralytics YOLOv8
- Architecture: YOLOv8n
- Accuracy: mAP@50 ≈ XX% (fill after training)
- Labels: YOLO-formatted bounding boxes
- Outputs: Annotated images, metrics, detection visuals


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

- Source: [Kidney Stone Dataset on Kaggle](https://www.kaggle.com/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection)
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

## 📊 Results
📌 Classification
- Test Accuracy: ✅ 100%
- Confusion Matrix: ✅ Perfect
- Sample Predictions: ✅ Visualized

📌 Object Detection (YOLOv8)
- Training Epochs: 50
- Model: YOLOv8n
- mAP@50:  0.85 (or your real number)
- Sample Detection:
  ![sample output](https://github.com/ahmadi-hossein/1--KidneyStone/blob/main/download.png)

## 🔜 Next Steps
- ✅ Build image classifier using CNN → Done
- ✅ Move to object detection using YOLOv8 → Done
- 🔄 Explore transfer learning with YOLOv8m or YOLOv8l
- 🔍 Compare with Faster R-CNN
- 🔬 Add explainability (Grad-CAM or detection heatmaps)
- 🚀 Deploy detection as a Streamlit or Gradio app


### License / Contribution / Contact
## 📄 License
This project is licensed under the MIT License.

## 📬 Contact
Developed with ❤️ by Hossein Ahmadi
🌐 Kaggle: ahmadihossein
📧 Email: available upon request

🌟 Star this project if you find it useful!
