import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # 1. Load Data
    file_path = 'Churn_Modelling.csv'
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # 2. Preprocessing
    print("\nOriginal Shape:", df.shape)
    
    # Drop irrelevant columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    print("Shape after dropping columns:", df.shape)
    
    # One-Hot Encoding
    print("Encoding categorical variables...")
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    
    # Define features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model Training & Evaluation
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"\nTraining {name}...")
        
        # Use scaled data for Logistic Regression, others can handle unscaled but scaled is fine
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"--- {name} Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[i])
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices_churn.png')
    print("\nConfusion matrices saved to confusion_matrices_churn.png")
    
    # 4. Feature Importance (Random Forest)
    print("\nGenerating Feature Importance Plot...")
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    feature_names = X.columns
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance saved to feature_importance.png")
    
    print("\nSummary of Accuracy:")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
    
    best_model = max(results, key=results.get)
    print(f"\nBest Model: {best_model}")

if __name__ == "__main__":
    main()
