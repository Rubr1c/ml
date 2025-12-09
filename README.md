# Chest X-Ray Pneumonia Classifier

A CNN-based binary classifier for detecting pneumonia from chest X-ray images using TensorFlow/Keras.

## Overview

This project implements a deep learning model to classify chest X-ray images as either **NORMAL** or **PNEUMONIA**. The model uses a Convolutional Neural Network (CNN) architecture and includes proper patient-wise data splitting to prevent data leakage.

## Dataset

The project uses the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle, containing 5,856 chest X-ray images.

- **Training set**: 4,081 images (1,111 normal, 2,970 pneumonia)
- **Validation set**: 850 images (231 normal, 619 pneumonia)
- **Test set**: 925 images (241 normal, 684 pneumonia)

Data is split by patient ID to ensure no patient appears in multiple splits.

## Features

- Patient-wise data splitting to prevent data leakage
- Data augmentation for improved model generalization
- Model evaluation with confusion matrices and ROC curves
- Visualization of training history and model performance

