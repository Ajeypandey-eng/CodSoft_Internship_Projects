import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load Dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv('Titanic-Dataset.csv')
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: 'Titanic-Dataset.csv' not found. Please ensure the file is in the same directory.")
        return

    # 2. Data Exploration & Preprocessing
    print("\nPre-processing data...")
    
    # Handle missing values
    # Age: Fill with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: Fill with mode
    if df['Embarked'].isnull().sum() > 0:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
    # Drop columns that are not useful for prediction or have too many missing values
    # Cabin has a lot of missing values. PassengerId, Name, Ticket are high cardinality identifiers.
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    
    # Encode Categorical Variables
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex']) # male: 1, female: 0
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    
    print("Data processed. Columns used:", df.columns.tolist())

    # 3. Model Implementation
    print("\nBuilding model...")
    
    # Split data into features (X) and target (y)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print("Model trained.")

    # 4. Evaluation
    print("\nEvaluating model...")
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance
    feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(feature_imp)

if __name__ == "__main__":
    main()
