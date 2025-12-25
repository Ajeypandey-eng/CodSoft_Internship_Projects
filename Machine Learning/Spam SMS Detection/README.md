# 📩 SMS Spam Detection

**Classifying SMS messages as Spam or Ham.**

This project uses NLP techniques to detect spam messages.

## 📊 Dataset
-   **Source**: `spam.csv`
-   **Structure**: `v1` (Label: ham/spam), `v2` (Message Text).

## 🛠️ Methodology
1.  **Preprocessing**: Text cleaning and TF-IDF Vectorization.
2.  **Models**:
    -   Multinomial Naive Bayes
    -   Logistic Regression
    -   Support Vector Machines (SVM)

## 📈 Results
-   **Naive Bayes** typically excels at this task due to the independence assumption working well with text tokens.
-   The model achieves high precision in flagging spam.

## 🚀 Usage
```bash
pip install -r requirements.txt
python spam_detection.py
```
