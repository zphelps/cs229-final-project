import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
    
    # Encode driverId, circuitId, constructorId
    cat_cols = ['driverId', 'circuitId', 'constructorId']
    encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col])
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le
    
    return train_df, test_df, encoders

def random_forest_podium_classifier(csv_path='cleaned/results2.csv'):
    # 1. Load data
    train_df, test_df, encoders = load_data(csv_path)
    
    # 2. Define feature columns
    feature_cols = [
        'driverId', 
        'circuitId', 
        'constructorId', 
        'round', 
        'grid', 
        # 'points_driver', 
        # 'points_constructor',
        'driver_points',
        'constructor_points',
        # 'driver_wins',
        # 'constructor_wins'
    ]
    
    # 3. Prepare the target: 1 if position in [1,2,3], else 0
    train_df['podium'] = train_df['position'].isin(["1","2","3"]).astype(int)
    test_df['podium'] = test_df['position'].isin(["1","2","3"]).astype(int)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['podium'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['podium'].values
    
    # 4. Train RandomForest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        random_state=42,       # For reproducibility
        max_depth=None,        # Let trees grow fully (could tune this)
        min_samples_split=2,   # Minimum samples to split an internal node
        n_jobs=-1              # Use all available CPU cores
    )
    rf_model.fit(X_train, y_train)
    
    # 5. Predict on test set
    y_pred = rf_model.predict(X_test)
    
    # 6. Compute and print metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("RandomForest Podium Classification Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    random_forest_podium_classifier()
