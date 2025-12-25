# 🖼️ AI Image Caption Generator

**"Show me a picture, and I'll tell you a story."**

This project implements a **Multimodal Neural Network** that automatically generates descriptive captions for images. It bridges the gap between **Computer Vision (CV)** and **Natural Language Processing (NLP)**.

---

## 🚀 How It Works

The model uses an **Encoder-Decoder** architecture:

1.  **The Eyes (Encoder):** Uses **VGG16** (pre-trained on ImageNet) to extract high-level visual features from the image. It converts an image into a 4,096-dimensional vector.
2.  **The Mouth (Decoder):** Uses an **LSTM (Long Short-Term Memory)** network to process the sequence of words. It takes the image vector and previous words to predict the next word in the sentence.

### Technical Stack
-   **Language:** Python
-   **Deep Learning:** TensorFlow / Keras
-   **Architecture:** CNN (VGG16) + RNN (LSTM)
-   **Dataset:** Flickr8k

---

## 📂 Project Structure

-   `caption_generator.py`: The main script. Handles data loading, preprocessing, model training, and inference.
-   `requirements.txt`: List of dependencies.
-   `README.md`: This file.

---

## 🛠️ Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place the **Flickr8k dataset** in the project folder:
-   `Images/`: Directory containing all .jpg images.
-   `captions.txt`: Text file with image-to-caption mappings.

### 3. Run the AI
```bash
python caption_generator.py
```
-   **First Run:** It will extract features from all 8,000 images (takes ~1 hour). Then it trains the model.
-   **Subsequent Runs:** It detects the saved model (`caption_model.h5`) and instantly generates captions for test images.

---

## 📊 Sample Results

The AI tries to "see" and describe the world:

> **Actual:** *A man prepares to enter the red building.*
> **Predicted:** *A man in a red jacket and jeans is standing on a white rail.*
> *(The AI correctly identified the "man" and "red", but hallucinated the rail!)*

> **Actual:** *A couple posing in front of a picture wall.*
> **Predicted:** *A man in a kilt and a woman standing in front of a church.*
> *(Correctly identified a "man", "woman", and "standing in front of" something.)*

---

## 🧠 Toddler Explanation
**Think of it like a robot:**
1.  The robot has **Magic Eyes (VGG16)** that turn pictures into ideas like "furry", "green", "running".
2.  The robot has a **Storyteller Mouth (LSTM)** that takes those ideas and makes a sentence: "A... dog... is... running."

---

*Project developed for CodSoft Internship.*
