# YCC-CovidCoughNet

A lightweight and modular deep learning pipeline for COVID-19 cough sound classification using **YAMNet**, **CNN**, and **Closed-form Continuous-time Neural Network (CFC)**.

## 🎯 Project Objective

To classify COVID-19 cough audio recordings (positive/negative) using a deep learning architecture optimized for **small datasets**, with an emphasis on:
- 🧠 Feature extraction with YAMNet (Google's pretrained audio model)
- 🔍 CNN for feature enhancement
- ⏱️ CFC for classification, offering time-continuous modeling without complex ODEs
- 🤖 Adam optimizer to improve training performance

## 🗂️ Project Structure

```bash
data/               # Raw and preprocessed cough audio files
features/           # Extracted audio features (YAMNet embeddings)
models/             # Saved models after training
utils/              # All training, preprocessing, and feature extraction code
notebooks/          # Optional Jupyter notebooks for testing
main.py             # Main execution script
requirements.txt    # Python dependencies
