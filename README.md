# 🏆 Kidney Stone Detection Challenge

Welcome to the Kidney Stone Detection Challenge!
In this project, you will work with a real-world medical imaging dataset to build deep learning models capable of detecting and localizing kidney stones in ultrasound images.

📂 Dataset Overview
The dataset comprises 9,416 ultrasound images categorized into two classes:

Normal: 4,414 images
Stone: 5,002 images
All images are resized to 512 x 512 pixels.
They were collected from multiple scan centers and hospitals using ultrasound machines such as SAMSUNG RS85, SAMSUNG HS60, SAMSUNG RS80A, and SAMSUNG HS70A, while ensuring the privacy and confidentiality of patient data.

License: CC BY 4.0

# 🧠 Objectives
You will complete two main tasks:

✅ Task 1: Image Classification

Develop a deep learning model to classify ultrasound images as either:

Normal
Stone

💡 Suggested Approach:

Use transfer learning with pre-trained CNN architectures (e.g., ResNet, EfficientNet).
Experiment with data augmentation to improve generalization.
Evaluate your model with accuracy, precision, recall, and confusion matrix.

Deliverables:

A Jupyter Notebook or Python script with the complete training pipeline.
A summary of results and evaluation metrics.

---------------------------------------------------------------------------------------------------

✅ Task 2: Object Detection of Kidney Stones

This task focuses only on the Stone class images. 

Steps:

Annotation:
Manually label the regions containing kidney stones using bounding boxes.
Tools you can use:
Label Studio
CVAT
Roboflow Annotate
Save annotations in YOLO format or Pascal VOC (XML).
Tip: Annotate carefully to capture the precise location and size of stones.

Model Development:
Train an object detection model (e.g., YOLOv8, YOLOv11, Faster R-CNN) to automatically locate kidney stones in new images.
Evaluate using mean Average Precision (mAP), IoU, and detection accuracy.

Deliverables:
Your labeled dataset (exported annotations).
A Jupyter Notebook or script demonstrating model training and inference.
Example predictions visualized on test images.


# 📘 Resources to Study
Before starting, you are encouraged to review:

Ultralytics Doc: [Click here](https://docs.ultralytics.com)
Radiopaedia Doc: [Click here](https://radiopaedia.org/articles/urolithiasis?lang=us)


# Best Wishes, Mr Momeni.
