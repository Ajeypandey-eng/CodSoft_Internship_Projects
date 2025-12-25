# Face Recognition Project

## Overview
This project contains two scripts:
1.  `face_detection.py`: A simple, fast face detector using OpenCV (Haar Cascades).
2.  `main_recognition.py`: A smart recognition system utilizing **DeepFace**.

## Setup
### 1. Install Dependencies
```bash
pip install opencv-python deepface tf-keras
```

### 2. Prepare Data
-   Create a folder named `known_faces`.
-   Add photos of people you want to recognize.
-   Name the files correctly (e.g., `Obama.jpg`, `Elon_Musk.jpg`). The script uses the filename as the label.

## Usage
### Run Face Detection (Basic)
```bash
python face_detection.py
```
*   Opens webcam.
*   Draws green boxes around faces.
*   Press 'q' to quit.

### Run Face Recognition (DeepFace)
```bash
python main_recognition.py
```
*   Loads images from `known_faces/`.
*   Scans the webcam feed.
*   Matches faces against your database and labels them.
*   **Note**: The first time you run this, it might download the VGG-Face model weights (~500MB).
