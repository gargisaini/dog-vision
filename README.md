# 🐶 Dog Breed Classification Using Transfer Learning

This project builds a deep learning model to classify **120 dog breeds** using the [Kaggle Dog Breed Identification Dataset](https://www.kaggle.com/c/dog-breed-identification). It leverages **Transfer Learning** with **MobileNetV2** pretrained on ImageNet, implemented using **TensorFlow 2.x**.

---

## 📌 Project Overview

- **Problem**: Multi-class image classification (120 dog breeds)
- **Dataset**: 10,000+ labeled dog images from Kaggle
- **Model**: MobileNetV2 (transfer learning)
- **Tools**: TensorFlow 2.x, Keras, TensorBoard, Matplotlib

---

## 🧠 Workflow Steps

### 📁 Data Preparation
- Downloaded and extracted the dataset from Kaggle.
- Loaded image IDs and breed labels from `labels.csv`.
- Mapped image IDs to breeds.
- Split dataset into **training**, **validation**, and **test** sets.
- Resized images to 224×224 and normalized pixel values.
- Used `tf.data` pipeline for efficient loading and batching.

### 🏗️ Model Building
- Used `tf.keras.applications.MobileNetV2` as the feature extractor.
- Replaced top layers with:
  - Global Average Pooling
  - Dense + Dropout
  - Final Dense layer with softmax activation (120 classes)
- Compiled with:
  - **Optimizer**: Adam
  - **Loss**: Categorical Crossentropy
  - **Metrics**: Accuracy

### 🏋️ Training
- Applied data augmentation (rotation, flip, zoom, etc.).
- Used callbacks:
  - EarlyStopping
  - ModelCheckpoint
  - TensorBoard logging
- Trained on 10,000+ images with efficient GPU usage.

### 📊 Evaluation
- Evaluated performance on validation and test sets.
- Visualized:
  - Accuracy & Loss curves
  - Confusion Matrix
  - Classification Report (precision, recall, F1-score)

### 📈 Improvements
- Started with a small subset for quick iterations.
- Scaled up to the full dataset (~10,000 images).
- Fine-tuned deeper layers of MobileNetV2 to improve accuracy.

### 💾 Saving & Exporting
- Saved the final model in:
  - `.h5` format (HDF5)
  - `SavedModel` format (for TensorFlow Serving)
- Demonstrated predictions on custom dog images.

---

