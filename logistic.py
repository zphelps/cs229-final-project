import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(csv_path):
    """
    Reads the CSV, splits into training and test sets (year==2024 for test),
    and applies LabelEncoding for categorical fields.
    
    Returns:
        train_df, test_df, encoders (dict of label encoders)
    """
    df = pd.read_csv(csv_path)
    
    # Split into train/test by year
    train_df = df[(df['year'] >= 2014) & (df['year'] <= 2023)].copy()
    test_df = df[df['year'] == 2024].copy()
    
    # We will encode driverId, circuitId, constructorId as categorical
    cat_cols = ['driverId', 'circuitId', 'constructorId']
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on full data for consistent mapping
        le.fit(df[col])
        
        # Transform both train and test
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        
        encoders[col] = le
    
    return train_df, test_df, encoders

def logistic_regression_baseline(csv_path='cleaned/results2.csv'):
    # 1. Load data
    train_df, test_df, encoders = load_data(csv_path)
    
    # 2. Define feature columns
    #    (Same ones used in the neural network: driverId, circuitId, constructorId, round, grid, points_driver, points_constructor)
    # feature_cols = ['driverId', 'circuitId', 'constructorId', 'round', 'grid', 'points_driver', 'points_constructor']
    feature_cols = [
        'driverId', 
        'circuitId', 
        'constructorId', 
        'round', 
        'grid', 
        'driver_points', 
        'constructor_points', 
        'driver_wins', 
        'constructor_wins',
    ]
    
    # 3. Prepare X (features) and y (labels)
    X_train = train_df[feature_cols].values
    y_train = train_df['position'].isin(["1","2","3"]).astype(int).values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['position'].isin(["1","2","3"]).astype(int).values
    
    # 4. Train logistic regression model
    lr_model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
    lr_model.fit(X_train, y_train)
    
    # 5. Predict on test set
    y_pred = lr_model.predict(X_test)
    
    # 6. Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # 7. Print results
    print("Logistic Regression Baseline Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    logistic_regression_baseline()
