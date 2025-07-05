# ✋ Gesture Recognition 🤖  
_Real-time Gesture Recognition using Rule-Based Logic and Random Forest Classifier_

![License](https://img.shields.io/badge/license-MIT-green)
![Language](https://img.shields.io/badge/Language-Python-blue)
![ML](https://img.shields.io/badge/Model-Random%20Forest%20%2B%20Rule%20Based-yellow)

---

## 📌 Overview

This project implements a hybrid **Gesture Recognition System** that uses both:

- ✅ **Rule-Based Logic** (using geometric hand landmark features)
- 🌲 **Random Forest Classifier** (trained on extracted keypoints)

The system processes input from a webcam or video feed to recognize predefined hand gestures in real time. It's designed for human-computer interaction, sign recognition, or touchless control.

---

## ✨ Features

- 📷 Real-time gesture detection using OpenCV
- ✋ Rule-based gesture classification for fast, interpretable results
- 🌲 Random Forest model trained on hand landmarks for higher accuracy
- 📊 Visual feedback with annotations on the video frame

---

## 🛠️ Tech Stack

| Component      | Tool/Library       |
|----------------|--------------------|
| Language       | Python             |
| Computer Vision| OpenCV             |
| ML Model       | Scikit-learn (Random Forest) |
| Landmark Detection | MediaPipe Hands (or custom code) |
| Visualization  | OpenCV, Matplotlib |
| Notebook       | Jupyter Notebook   |

---

## How  It Works

Rule-Based Mode: Uses angle, distance, and position between fingers to classify gestures (e.g., open palm, fist, peace sign).
Random Forest Mode: Trained on hand landmark coordinates; generalizes better across noisy inputs or custom gestures.
You can toggle between the two approaches depending on use case and performance.

## Recognized Gestures

✊ Fist
✌️ Victory / Peace
👍 Thumbs Up
🤟 Rockstar
🤙 YOLO
