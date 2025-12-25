# 🚢 Titanic Survival Prediction

**Predicting who survived the Titanic disaster.**

This classic Data Science project uses passenger data to predict survival outcomes.

## 📊 Dataset
-   **Source**: `Titanic-Dataset.csv`
-   **Features**: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

## 🛠️ Methodology
1.  **Data Cleaning**: Hadling missing values in Age and Embarked columns.
2.  **Feature Engineering**: Encoding categorical variables (Sex, Embarked).
3.  **Modeling**:
    -   Logistic Regression
    -   Decision Tree Classifier
    -   Random Forest Classifier

## 📈 Results
-   **Gender (Sex)** is the most significant predictor of survival.
-   **Random Forest** achieved the highest accuracy on the test set.

## 🚀 Usage
```bash
pip install -r requirements.txt
python Titanic.py
```
