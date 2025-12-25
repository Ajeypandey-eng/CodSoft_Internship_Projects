import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(path):
    print(f"Loading {path}...")
    try:
        # Check encoding or use 'python' engine for multi-char separator if needed
        return pd.read_csv(path, sep=':::', engine='python', header=None, names=['ID', 'Title', 'Genre', 'Description'])
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # 1. Load Data
    train_path = 'train_data.txt'
    test_path = 'test_data_solution.txt'
    
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    
    if train_df is None or test_df is None:
        return

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)
    
    # 2. Preprocessing
    print("\nCleaning text...")
    train_df['clean_desc'] = train_df['Description'].apply(clean_text)
    test_df['clean_desc'] = test_df['Description'].apply(clean_text)
    
    print("\nTop 5 Genres in Training Data:")
    print(train_df['Genre'].value_counts().head())
    
    # 3. Vectorization
    print("\nVectorizing text (TF-IDF, max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = tfidf.fit_transform(train_df['clean_desc'])
    X_test = tfidf.transform(test_df['clean_desc'])
    
    y_train = train_df['Genre'].str.strip() # Remove potential whitespace
    y_test = test_df['Genre'].str.strip()
    
    # 4. Model Training & Evaluation
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVC": LinearSVC(random_state=42, dual='auto')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"--- {name} Results ---")
        print(f"Accuracy: {acc:.4f}")
        # Only print classification report for top genres to avoid huge output
        print(classification_report(y_test, y_pred, zero_division=0))
        
        if name == "Linear SVC": # Save confusion matrix for the likely best model
            # Filter for top 10 genres for cleaner plot
            top_genres = y_test.value_counts().head(10).index
            mask = y_test.isin(top_genres)
            
            # Re-predict only for these samples for the plot
            y_test_top = y_test[mask]
            y_pred_top = pd.Series(y_pred)[mask.reset_index(drop=True)] # handle index mismatch
            
            cm = confusion_matrix(y_test_top, y_pred_top, labels=top_genres)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=top_genres, yticklabels=top_genres)
            plt.title('Confusion Matrix (Top 10 Genres) - Linear SVC')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig('confusion_matrix_svc.png')
            print("Confusion matrix for Top 10 Genres saved to confusion_matrix_svc.png")

    print("\nSummary of Accuracy:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

    best_model = max(results, key=results.get)
    print(f"\nBest Model: {best_model}")

if __name__ == "__main__":
    main()
