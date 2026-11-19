# Brain Tumor MRI Classification System

An end-to-end deep learning web application for classifying **T1-weighted brain MRI scans** into four categories: **Glioma**, **Meningioma**, **Pituitary tumor**, and **No Tumor**.

**Live Demo:** [braintumoradvisoryapp.streamlit.app](https://braintumoradvisoryapp.streamlit.app)<br>
*If the app is inactive, click the blue button to wake it up. Usually takes about 1 minute.*

## Overview

This repository contains the **monolithic deployment** of the `BrainTumorCNN` model, combining the PyTorch inference engine and the interactive user interface into a single Streamlit application optimized for **Streamlit Community Cloud**.

Users can upload one or multiple MRI scans (`.jpg`, `.png`) and receive instant classification results along with prediction confidence scores.

## Key Engineering Highlights

* **Custom 5-Block CNN:** Lightweight architecture built in PyTorch (`Conv2d` → `BatchNorm2d` → `ReLU` → `MaxPool2d`, followed by `AdaptiveAvgPool2d`, `Dropout(0.3)`, and a linear classifier).

* **Eliminating Shortcut Learning:** Guided by **Grad-CAM** heatmaps, custom preprocessing transforms (`CropBlackSpace` for automatic skull extraction and Lanczos-based `PadToSize` to `512×512`) were engineered so the model focuses strictly on brain tissue rather than background artifacts.

* **Leak-Free Data Pipeline:** Trained on combined **Figshare** + **Br35H** datasets with patient-ID splitting, `difPy` deduplication, and weighted `CrossEntropyLoss` to handle class imbalance.

* **Optimized Inference:** Runs in `model.eval()` mode with `torch.no_grad()` for fast, low-memory CPU inference in the cloud.

## Model Performance

Evaluated on a fully isolated external test set (**Sartaj**, 2,832 deduplicated scans):

| Metric | Overall (Weighted) | 
| ----- | ----- | 
| **Accuracy** | **83.65%** | 
| **Precision** | **84.20%** | 
| **Recall** | **83.65%** | 
| **F1-Score** | **83.73%** | 

| Class | Precision | Recall | F1-Score | Support | 
| ----- | ----- | ----- | ----- | ----- | 
| **Glioma** | 87.41% | 84.67% | 86.02% | 861 | 
| **Meningioma** | 83.45% | 74.07% | 78.48% | 810 | 
| **No Tumor** | 61.39% | 76.07% | 67.95% | 326 | 
| **Pituitary** | 90.56% | 94.93% | 92.70% | 829 | 


## Disclaimer

This application was developed as an Engineering Thesis project for **educational and research purposes only**. It is not a certified medical device and must not be used for clinical diagnosis.