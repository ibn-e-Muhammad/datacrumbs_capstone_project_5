import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def preprocess_data(file_path):
    print("--- Phase 2 EDA Metrics ---")
    
    # 1. Load dataset
    df = pd.read_csv(file_path)
    print(f"Original Dataset Shape: {df.shape}")
    
    # 2. Handle zero values in biological columns
    biological_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print("\nMissing/Zero values handled:")
    for col in biological_cols:
        zero_count = (df[col] == 0).sum()
        print(f" - {col}: {zero_count} zero values replaced with median.")
        # Replace 0 with NaN to compute accurate median
        df[col] = df[col].replace(0, np.nan)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
    print("\nClass Distribution:")
    print(df['Outcome'].value_counts().to_string())

    # 3. Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # 4. Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain set shape (X, y): {X_train.shape}, {y_train.shape}")
    print(f"Test set shape (X, y): {X_test.shape}, {y_test.shape}")
    
    # 5. Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Save the objects
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    np.save(os.path.join(base_dir, 'data', 'processed', 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(base_dir, 'data', 'processed', 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(base_dir, 'data', 'processed', 'y_train.npy'), y_train.to_numpy())
    np.save(os.path.join(base_dir, 'data', 'processed', 'y_test.npy'), y_test.to_numpy())
    
    with open(os.path.join(base_dir, 'models', 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print("\nProcessing complete. Artifacts saved to data/processed/ and models/ directories.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocess_data(os.path.join(base_dir, 'diabetes.csv'))
