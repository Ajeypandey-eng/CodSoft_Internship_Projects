# Cognitive Computing & Artificial Intelligence Portfolio

This repository features a collection of projects focused on **Cognitive Computing**, simulating human thought processes through **Neural Networks**, **Statistical Reasoning**, and **Natural Language Processing**. These systems are designed to perceive, reason, and learn from complex, unstructured data.

## 📂 Project Directory

### 🧠 Language & Semantic Understanding (NLP)
1.  **[Image Captioning](./Artificial%20Intelligence/Image%20Captioning/)**: A multimodal system combining **VGG16 and LSTM** to perceive visual content and generate human-like descriptive natural language.
2.  **[Movie Genre Classification](./Machine%20Learning/Movie%20Genre%20Classification/)**: An NLP-driven reasoning model that analyzes semantic themes in plot summaries to categorize cinematic genres using **TF-IDF Vectorization**.
3.  **[SMS Spam Detection](./Machine%20Learning/Spam%20SMS%20Detection/)**: A text-classification engine using **Naive Bayes** to identify and filter linguistic patterns associated with fraudulent messaging.

### 👁️ Visual Intelligence & Perception
1.  **[Face Recognition](./Artificial%20Intelligence/Face%20Recognition/)**: A real-time biometric perception system leveraging **DeepFace** (VGG-Face/Facenet) for high-accuracy human identity verification and facial feature analysis.
2.  **[Tic-Tac-Toe AI](./Artificial%20Intelligence/Tic%20Tac%20Toe%20AI/)**: A decision-making agent using the **Minimax algorithm** to simulate strategic human reasoning and optimal game-play through recursive state-space exploration.

### 📊 Advanced Predictive Modeling & Reasoning
1.  **[Customer Churn Prediction](./Machine%20Learning/Customer%20Churn%20Prediction/)**: A behavioral analysis model using **Gradient Boosting** to forecast customer attrition and assist in retention decision-making.
2.  **[Credit Card Fraud Detection](./Machine%20Learning/Credit%20Card%20Problem/)**: An anomaly detection system utilizing **SMOTE** (for class imbalance) and **Random Forest** to recognize cognitive patterns of fraudulent financial behavior.
3.  **[Sales Prediction](./Data%20Science/Sales%20Prediction/)**: A regression-based forecasting model that predicts revenue trends by analyzing the impact of multi-channel advertising spend.
4.  **[Titanic Survival Prediction](./Data%20Science/Titanic%20Problem/)**: A classification project exploring historical data to reason through the socio-economic factors that influenced survival probabilities.

---

## 🛠️ Tech Stack & Core Libraries

The following technologies form the backbone of these cognitive systems:

| Category | Tools & Libraries |
| :--- | :--- |
| **Deep Learning** | TensorFlow, Keras, DeepFace, VGG16, LSTM |
| **Machine Learning** | Scikit-Learn, XGBoost, Random Forest, Naive Bayes |
| **Data Manipulation** | Pandas, NumPy, SMOTE (Imbalanced-Learn) |
| **NLP & Vision** | NLTK, OpenCV, TF-IDF, Matplotlib, Seaborn |

---

## 🌟 Project Spotlight: Image Captioning Engine

This project represents the pinnacle of cognitive simulation within this portfolio, where the model performs **visual-to-semantic translation**.

### The Challenge
How can a machine describe a scene it has never seen before? Unlike simple classification (identifying "a dog"), captioning requires understanding relationships between objects, actions, and context (identifying "a dog running through a grassy field").

### Technical Architecture: The Encoder-Decoder Framework
The system utilizes a hybrid neural network architecture:
* **The Encoder (VGG16)**: A pre-trained Convolutional Neural Network (CNN) that "sees" the image. It extracts complex mathematical features from the pixels, effectively acting as the model's visual cortex.
* **The Decoder (LSTM)**: A Long Short-Term Memory network that "speaks." It takes the visual features and generates a sequence of words, one by one, using stateful memory to ensure the sentence is grammatically coherent.

### Key Learnings
* **Feature Extraction**: Learning how to strip the final classification layer of a CNN to use it as a feature vector.
* **Sequence Modeling**: Managing variable-length text data and word embeddings.
* **Multimodal Fusion**: Merging two distinct types of data (spatial/images and sequential/text) into a single learning pipeline.

---

## 🚀 Getting Started

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/Ajeypandey-eng/Cognitive-Computing-Projects.git](https://github.com/Ajeypandey-eng/Cognitive-Computing-Projects.git)
    cd Cognitive-Computing-Projects
    ```
2.  **Setup Environment**:
    It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### Execution
Navigate to any specific project folder to find a dedicated `README.md` with execution details for that specific cognitive model. For example:
```bash
cd "Artificial Intelligence/Face Recognition"
python main.py
