# 📈 Sales Prediction

**Predicting sales based on advertising spend.**

This project builds a regression model to predict sales revenue based on advertising budgets for TV, Radio, and Newspaper.

## 📊 Dataset
-   **Source**: `advertising.csv`
-   **Features**: `TV`, `Radio`, `Newspaper` (Ad budgets).
-   **Target**: `Sales`.

## 🛠️ Methodology
-   **Exploratory Data Analysis (EDA)**: Correlation heatmaps and pair plots to understand relationships.
-   **Models**:
    -   Linear Regression
    -   Random Forest Regressor

## 📈 Results
-   **TV advertising** has the strongest correlation with Sales.
-   The model can accurately forecast sales given a marketing budget plan.

## 🚀 Usage
```bash
pip install -r requirements.txt
python sales_prediction.py
```
