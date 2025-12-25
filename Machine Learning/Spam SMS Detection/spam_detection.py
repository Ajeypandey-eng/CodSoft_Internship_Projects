import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def main():
    # 1. Load Data
    file_path = 'spam.csv'
    print(f"Loading data from {file_path}...")
    try:
        # formatting issues with this dataset often require latin-1
        df = pd.read_csv(file_path, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # 2. Preprocessing
    # Keep only relevant columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    print("\nData loaded. Shape:", df.shape)
    print("Class Distribution:\n", df['label'].value_counts())
    
    # Encode labels
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Clean text
    print("Cleaning text...")
    df['clean_message'] = df['message'].apply(clean_text)
    
    # 3. Vectorization (TF-IDF)
    print("Vectorizing text (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    X = tfidf.fit_transform(df['clean_message']).toarray()
    y = df['label_enc']
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Model Training & Evaluation
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42)
    }
    
    results = {}
    
    # Prepare plot for Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"--- {name} Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("\nConfusion matrices saved to confusion_matrices.png")
    
    print("\nSummary of Accuracy:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

if __name__ == "__main__":
    main()
