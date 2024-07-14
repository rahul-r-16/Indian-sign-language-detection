# Indian Sign Language Detection

## Overview

This repository contains code for a deep learning model designed to detect Indian Sign Language (ISL) gestures in real-time using TensorFlow/Keras.

### Features

- **Data Preprocessing**: Images are preprocessed using OpenCV for resizing, normalization, and augmentation.
- **Model Architecture**: Utilizes a Convolutional Neural Network (CNN) for feature extraction and classification.
- **Training**: The model is trained on a dataset containing hand gesture images corresponding to 24 different ISL signs.
- **Evaluation**: Performance metrics such as accuracy and loss are monitored during training and evaluated on a separate test dataset.
- **Deployment**: Instructions for deploying the trained model for real-time inference.

## Dataset

The dataset consists of images categorized into train and test sets for training and validation purposes. It includes various hand gestures from the Indian Sign Language.

## Installation

To run the code, ensure you have Python 3.x installed along with the following dependencies:

```bash
pip install tensorflow opencv-python matplotlib
