import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import sys

# Try to import imblearn, if not present handle gracefully (though we plan to use it)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: imbalanced-learn not installed. SMOTE will be skipped, and we will use class weights or simple undersampling if needed.")

def main():
    # 1. Load Dataset
    file_path = 'creditcard.csv'
    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # 2. Exploratory Analysis & Preprocessing
    print("\nData Loaded. Shape:", df.shape)
    print("Class Distribution:\n", df['Class'].value_counts())
    
    # Check for nulls
    if df.isnull().sum().max() > 0:
        print("Handling missing values...")
        df = df.dropna()

    # Scale 'Amount'
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Drop 'Time' and original 'Amount'
    df = df.drop(['Time', 'Amount'], axis=1)
    
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Split Data
    print("\nSplitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 4. Handle Imbalance
    if HAS_SMOTE:
        print("\nApplying SMOTE to training data...")
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print("Resampled Training set shape:", X_train_res.shape)
        print("Resampled Class Distribution:\n", y_train_res.value_counts())
    else:
        print("\nSMOTE not available. Using original training data (imbalanced).")
        X_train_res, y_train_res = X_train, y_train

    # 5. Model Training & Evaluation
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTRAINING {name}...")
        model.fit(X_train_res, y_train_res)
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        print(f"\n--- {name} Results ---")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        results[name] = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

    print("\nSummary of F1-Scores:")
    for name, metrics in results.items():
        print(f"{name}: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
