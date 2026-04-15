# Pet Image Segmentation using U-Net & TensorFlow

## Overview
This project implements an image segmentation model for pets using deep learning. The goal is to classify each pixel in an image into different categories (e.g., pet, background, outline) using a U-Net architecture with a MobileNetV2 backbone.

The model is trained on the Oxford-IIIT Pet Dataset, which contains annotated images of cats and dogs with segmentation masks.

## Features
End-to-end image segmentation pipeline
Uses TensorFlow & tf.data API for efficient data loading
Transfer learning with MobileNetV2 as encoder
Data augmentation support
Visualization of predictions
Optional class-weighted training to handle imbalance
Classes:
Pet    Background    Outline

## Tech Stack
Python
TensorFlow / Keras
TensorFlow Datasets (TFDS)
NumPy
Matplotlib

## Model Architecture
Encoder (Downsampling)
Pretrained MobileNetV2
Extracts feature maps at multiple scales
Decoder (Upsampling)
Uses pix2pix upsampling blocks
Skip connections from encoder (U-Net structure)
Output:
Pixel-wise classification with 3 output channels

## Key Learnings
Efficient use of tf.data pipelines
Transfer learning for segmentation
U-Net architecture design
Handling class imbalance with weighted loss
