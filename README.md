# 🐟 Practical Deep Learning – Fish Species Classification

This project implements and evaluates deep learning models for **multi-class fish species classification** using a large-scale image dataset. The work compares **custom CNN architectures** with **state-of-the-art transfer learning models**, and explores optimization techniques to achieve high accuracy and robustness.

---

## 📌 Project Overview

The goal of this project is to classify fish species from RGB images using deep learning techniques.  
The pipeline includes data preprocessing, augmentation, model training, evaluation, and optimization using both **custom CNNs** and **pre-trained architectures**.

Key highlights:
- Custom CNN trained with **K-Fold Cross-Validation**
- Fine-tuning multiple **transfer learning models**
- Inference-Time Augmentation (TTA)
- Hybrid **Deep Feature Extraction + SVM** approach
- Achieved **near-perfect accuracy (up to 100%)**

---

## 📊 Dataset

- **Total images:** ~9,000 (extended to ~9,456 with Salmon class)
- **Classes:** 9 → 10 fish species
- **Image format:** RGB
- **Original size:** 590×445
- **Resized to:** 160×160
- **Class balance:** Perfectly balanced (1,000 images per class)

### Preprocessing & Augmentation
- Normalization to [0,1]
- Rotation & flipping
- Zoom & shear transformations

---

## 🧠 Models & Methodology

### Custom CNN
- Trained using **5-Fold Cross-Validation**
- Batch Normalization & Learning Rate Scheduling
- Mean validation accuracy: **~99.4%**

### Transfer Learning
Fine-tuned pre-trained models:
- ResNet50
- VGG16
- DenseNet121
- EfficientNetB0

### Hybrid Model (Best Result)
- ResNet50 used as a **feature extractor**
- Feature vectors classified using **SVM**
- **Test Accuracy: 100%**

---

## 📈 Optimization Techniques
- Data augmentation
- ReduceLROnPlateau scheduler
- Batch normalization
- Inference-Time Augmentation (TTA)

---

## 🛠 Tech Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Models:** CNN, ResNet50, VGG16, DenseNet121, EfficientNetB0  
- **ML:** Scikit-learn (SVM)  
- **Data Processing:** NumPy, Pandas  
- **Evaluation:** K-Fold CV, Accuracy, Loss  
- **Tools:** Jupyter Notebook, Git  

---

## 📂 Repository Structure

```
├── ex1.ipynb
├── kfold_tta_results.csv
├── result/
├── README.md
├── license.txt
```

---

## 🏁 Results Summary

- Custom CNN: **~99.4% validation accuracy**
- Best Transfer Learning: **~99.9% accuracy**
- ResNet50 + SVM: **100% test accuracy**
- EfficientNetB0 offered best speed-accuracy tradeoff

---

## 👤 Author

**Dania Dabbah**  
Software & Information Systems Engineering Student  
Ben-Gurion University of the Negev  
GitHub: https://github.com/Dania2703
