# Breast Cancer Detection using Deep Learning

## Overview
This project implements an end-to-end **deep learning–based breast cancer detection system** using medical imaging data. The objective is to assist healthcare professionals by improving early detection accuracy and reducing diagnostic errors through automated image classification.

The system classifies breast images into **benign** and **malignant** categories using a Convolutional Neural Network (CNN) with transfer learning.

---
## Academic Context
This project was developed as part of my **B.Tech Final Year Project** and has been further structured and documented for professional and industry-oriented use.

---
## Problem Statement
Breast cancer diagnosis using traditional imaging techniques is time-consuming and subject to human interpretation errors. Early detection is critical for improving survival rates. This project leverages **deep learning and computer vision** to automate breast cancer detection from medical images, providing a scalable and reliable diagnostic support system.

---

## Tech Stack
- **Programming Language:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Data Handling:** NumPy, Pandas  
- **Image Processing:** OpenCV  
- **Visualization:** Matplotlib, Seaborn  
- **Web Framework:** Flask  
- **Version Control:** Git, GitHub  

---

```
## Project Structure

breast-cancer-detection-using-deep-learning/
│
├── app/ # (Flask application)
├── src/ # Model and testing scripts
├── data/ # Dataset Info(no raw data included)
├── models/ # Model info(weights excluded)
├── results/ # Evaluation images and outputs
├── templates/ # Frontend templates
├── docs/ # Project report
├── requirements.txt # Dependencies
├── .gitignore # Ignored files and folders
└── README.md
```
## Model Architecture
- **CNN with VGG16** used for feature extraction
- Transfer learning applied to improve performance on limited medical data
- Custom dense layers added for classification

## Evaluation Metrics
The model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve
**Achieved Accuracy:** ~96%

## Disclaimer
This project is intended for educational and research purposes only.
It is not a substitute for professional medical diagnosis.

## Author
Dev Patel - Data Science
