# 📉 Customer Churn Prediction

**Predicting whether a customer will leave the bank.**

This project uses Machine Learning to predict customer churn based on historical data. By analyzing factors like credit score, geography, gender, age, and balance, the model identifies customers at risk of exiting.

## 📊 Dataset
-   **Source**: `Churn_Modelling.csv`
-   **Features**: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
-   **Target**: `Exited` (1 = Churn, 0 = Stay).

## 🛠️ Models Implemented
-   Logistic Regression
-   Random Forest Classifier
-   Gradient Boosting Classifier

## 📈 Results
-   **Gradient Boosting** achieved the best performance with ~87% accuracy.
-   Key indicators of churn: **Age**, **Estimated Salary**, and **Balance**.

## 🚀 Usage
```bash
pip install -r requirements.txt
python churn_prediction.py
```
