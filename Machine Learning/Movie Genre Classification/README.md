# 🎬 Movie Genre Classification

**Predicting movie genres from plot summaries.**

This project builds a Natural Language Processing (NLP) model to classify movies into genres based on their descriptions.

## 📊 Dataset
-   **Files**: `train_data.txt`, `test_data.txt`
-   **Features**: Movie Description (Text)
-   **Target**: Genre (Drama, Thriller, Action, etc.)

## 🛠️ Methodology
1.  **Text Preprocessing**: Cleaning text (lowercased, removed punctuation).
2.  **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numbers.
3.  **Modeling**:
    -   Naive Bayes
    -   Logistic Regression
    -   Linear SVC

## 📈 Results
-   **Logistic Regression** performed best (~58% accuracy).
-   Challenges included handling class imbalance and overlapping genre definitions.

## 🚀 Usage
```bash
pip install -r requirements.txt
python genre_classification.py
```
