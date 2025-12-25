import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # 1. Load Data
    file_path = 'advertising.csv'
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # 2. EDA & Preprocessing
    print("\nData Shape:", df.shape)
    print("First 5 rows:\n", df.head())
    
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    print("\nCorrelation Matrix:")
    corr = df.corr()
    print(corr)
    
    # Simple Visualization (saved to file)
    print("\nGenerating correlation heatmap (correlation_heatmap.png)...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_heatmap.png')
    plt.close()

    print("Generating feature scatter plots (sales_vs_features.png)...")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    features = ['TV', 'Radio', 'Newspaper']
    for i, feature in enumerate(features):
        axs[i].scatter(df[feature], df['Sales'])
        axs[i].set_xlabel(feature)
        axs[i].set_ylabel('Sales')
        axs[i].set_title(f'Sales vs {feature}')
    plt.tight_layout()
    plt.savefig('sales_vs_features.png')
    plt.close()

    # 3. Split Data
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    
    print("\nSplitting data into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Model Training & Evaluation
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"R2": r2, "RMSE": rmse}
        
        print(f"--- {name} Performance ---")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        
        if name == "Linear Regression":
            print("Coefficients:", dict(zip(X.columns, model.coef_)))

    # 5. Conclusion
    print("\nSummary of R2 Scores:")
    for name, metrics in results.items():
        print(f"{name}: {metrics['R2']:.4f}")
        
    best_model = max(results, key=lambda x: results[x]['R2'])
    print(f"\nBest Model: {best_model}")

if __name__ == "__main__":
    main()
