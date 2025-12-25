# 💳 Credit Card Fraud Detection

**Detecting fraudulent transactions using Machine Learning.**

This project deals with the highly imbalanced dataset of credit card transactions to identify frauds. It employs techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance.

## 📊 Dataset
-   **Source**: `creditcard.csv` (Kaggle).
-   **Size**: ~284,807 transactions.
-   **Imbalance**: Only 0.17% of transactions are fraudulent.

## 🛠️ Methodology
1.  **Preprocessing**: Scaling 'Amount' and 'Time' features.
2.  **Addressing Imbalance**: Used **SMOTE** to oversample the minority (fraud) class.
3.  **Models**:
    -   Logistic Regression
    -   Random Forest Classifier

## 📈 Results
-   **Random Forest** provided superior precision and recall compared to Logistic Regression.
-   The model can effectively catch fraud while minimizing false alarms.

## 🚀 Usage
```bash
pip install -r requirements.txt
python fraud_detection.py
```
*(Note: You need to download `creditcard.csv` and place it in this folder if it's not present, as it is excluded from git).*
