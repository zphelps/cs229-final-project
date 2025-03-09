import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define our target variable based on position
# 1st, 2nd, 3rd and off podium (4+)
def create_target(position):
    if position == 1:
        return 'first'
    elif position == 2:
        return 'second'
    elif position == 3:
        return 'third'
    else:
        return 'off_podium'


def apply_labels(df):
    df['label'] = df['position'].apply(create_target)

    # Drop the position column
    df = df.drop('position', axis=1)

    return df


def main():
    # Load the data from the new CSV file
    df = pd.read_csv("../cleaned/f1_podium_prediction_dataset.csv")

    # Apply labels to the data
    df = apply_labels(df)
    
    # Drop columns that shouldn't be used for training
    # - podium: this is directly related to our target
    # - driver_name and constructor_name: these are text identifiers
    columns_to_drop = ['podium', 'driver_name', 'constructor_name', 'points']
    df = df.drop(columns_to_drop, axis=1)

    # Split data into training (before 2024) and testing (2024)
    train_df = df[(df['year'] < 2024) & (df['year'] >= 2000)]
    test_df = df[df['year'] == 2024]

    # Define categorical columns for one-hot encoding
    categorical_cols = ['driverId', 'constructorId', 'circuitId']
    
    # Define numerical columns (all columns except categorical and label)
    numerical_cols = [col for col in train_df.columns if col not in categorical_cols + ['label']]
    
    # Create preprocessing pipeline with one-hot encoding for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Apply preprocessing to training data
    X_train = train_df.drop(['label'], axis=1)
    y_train = train_df['label']
    
    # Apply preprocessing to test data
    X_test = test_df.drop(['label'], axis=1)
    y_test = test_df['label']

    print(X_train.columns)
    print(X_train.head())
    
    # Encode string labels to numeric values
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Store the mapping for later reference
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print("Label mapping:", label_mapping)
    
    # Create and train the XGBoost model with preprocessing
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42
        ))
    ])
    
    # Fit the model with encoded labels
    model.fit(X_train, y_train_encoded)

    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)
    
    # Map class indices to original labels
    first_idx = np.where(label_encoder.classes_ == 'first')[0][0]
    second_idx = np.where(label_encoder.classes_ == 'second')[0][0]
    third_idx = np.where(label_encoder.classes_ == 'third')[0][0]
    
    # Add probabilities back to the test dataframe for analysis
    test_with_proba = X_test.copy()
    test_with_proba['true_label'] = y_test
    test_with_proba['prob_first'] = y_proba[:, first_idx]
    test_with_proba['prob_second'] = y_proba[:, second_idx]
    test_with_proba['prob_third'] = y_proba[:, third_idx]
    
    # Group by race (raceId)
    race_groups = test_with_proba.groupby('raceId')
    
    print("\nPredicted vs Actual Podiums for Each Race (XGBoost):")
    print("=" * 80)

    # Initialize metrics
    total_races = 0
    correct_first = 0
    correct_second = 0
    correct_third = 0
    correct_podium_drivers = 0
    total_podium_drivers = 0
    position_errors = []
    complete_podiums = 0
    weighted_score = 0
    max_weighted_score = 0
    
    for race_id, race_data in race_groups:
        print(f"\nRace ID: {race_id}")
        total_races += 1
        
        # Get actual podium for this race
        actual_podium = race_data[race_data['true_label'].isin(['first', 'second', 'third'])].sort_values('true_label')
        
        # Create dictionaries for actual podium positions
        actual_positions = {}
        for _, row in actual_podium.iterrows():
            if row['true_label'] == 'first':
                actual_positions[row['driverId']] = 1
            elif row['true_label'] == 'second':
                actual_positions[row['driverId']] = 2
            elif row['true_label'] == 'third':
                actual_positions[row['driverId']] = 3
        
        # Print actual podium
        print("Actual Podium:")
        if len(actual_podium) > 0:
            for _, row in actual_podium.iterrows():
                print(f"  {row['true_label'].capitalize()}: Driver {row['driverId']}")
        else:
            print("  No podium data available for this race")
            continue  # Skip this race for metrics if no actual podium data
        
        # Create a copy of race data for predictions
        race_pred = race_data.copy()
        predicted_podium = []
        
        # Find driver with highest probability for first place
        if len(race_pred) > 0:
            first_place_idx = race_pred['prob_first'].idxmax()
            first_place_driver = race_pred.loc[first_place_idx]
            predicted_podium.append(('first', first_place_driver['driverId'], first_place_driver['constructorId']))
            
            # Remove the selected driver from consideration
            race_pred = race_pred.drop(first_place_idx)
            
            # Find driver with highest probability for second place
            if len(race_pred) > 0:
                second_place_idx = race_pred['prob_second'].idxmax()
                second_place_driver = race_pred.loc[second_place_idx]
                predicted_podium.append(('second', second_place_driver['driverId'], second_place_driver['constructorId']))
                
                # Remove the selected driver from consideration
                race_pred = race_pred.drop(second_place_idx)
                
                # Find driver with highest probability for third place
                if len(race_pred) > 0:
                    third_place_idx = race_pred['prob_third'].idxmax()
                    third_place_driver = race_pred.loc[third_place_idx]
                    predicted_podium.append(('third', third_place_driver['driverId'], third_place_driver['constructorId']))
        
        # Print predicted podium
        print("Predicted Podium:")
        for position, driver_id, constructor_id in predicted_podium:
            print(f"  {position.capitalize()}: Driver {driver_id} (Constructor {constructor_id})")
        
        # Calculate metrics for this race
        race_correct_podium = 0
        race_weighted_score = 0
        
        for i, (position, driver_id, _) in enumerate(predicted_podium):
            pred_pos = i + 1  # 1, 2, or 3
            
            # Position-specific accuracy
            if driver_id in actual_positions:
                actual_pos = actual_positions[driver_id]
                
                # Driver is on the podium (regardless of position)
                correct_podium_drivers += 1
                
                # Position-specific correctness
                if pred_pos == actual_pos:
                    if pred_pos == 1:
                        correct_first += 1
                        race_weighted_score += 3
                    elif pred_pos == 2:
                        correct_second += 1
                        race_weighted_score += 2
                    elif pred_pos == 3:
                        correct_third += 1
                        race_weighted_score += 1
                    
                    race_correct_podium += 1
                
                # Calculate positional distance error
                position_errors.append(abs(pred_pos - actual_pos))
            
            total_podium_drivers += 1
        
        # Complete podium accuracy
        if race_correct_podium == 3:
            complete_podiums += 1
        
        # Add to weighted score
        weighted_score += race_weighted_score
        max_weighted_score += 6  # Maximum possible score per race (3+2+1)
    
    # Calculate and print final metrics
    print("\n" + "=" * 80)
    print("\nCustom F1 Podium Prediction Metrics (XGBoost):")
    
    # Position-specific accuracy
    print(f"\nPosition-Specific Accuracy:")
    print(f"  1st Place: {correct_first/total_races:.4f} ({correct_first}/{total_races})")
    print(f"  2nd Place: {correct_second/total_races:.4f} ({correct_second}/{total_races})")
    print(f"  3rd Place: {correct_third/total_races:.4f} ({correct_third}/{total_races})")
    
    # Podium inclusion accuracy
    podium_inclusion = correct_podium_drivers/total_podium_drivers if total_podium_drivers > 0 else 0
    print(f"\nPodium Inclusion Accuracy: {podium_inclusion:.4f} ({correct_podium_drivers}/{total_podium_drivers})")
    
    # Mean positional distance error
    mean_pos_error = sum(position_errors)/len(position_errors) if position_errors else 0
    print(f"\nMean Positional Distance Error: {mean_pos_error:.4f}")
    
    # Complete podium accuracy
    complete_podium_acc = complete_podiums/total_races
    print(f"\nComplete Podium Accuracy: {complete_podium_acc:.4f} ({complete_podiums}/{total_races})")
    
    # Weighted podium score
    weighted_podium_score = weighted_score/max_weighted_score if max_weighted_score > 0 else 0
    print(f"\nWeighted Podium Score: {weighted_podium_score:.4f} ({weighted_score}/{max_weighted_score})")
    
    # Print feature importances
    if hasattr(model['classifier'], 'feature_importances_'):
        feature_names = (
            numerical_cols + 
            model['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols).tolist()
        )
        importances = model['classifier'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 15 Most Important Features:")
        for i in range(min(15, len(feature_names))):
            idx = indices[i]
            if idx < len(feature_names):
                print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Standard evaluation metrics on the original predictions
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    print("\n" + "=" * 80)
    print("\nStandard Classification Metrics (for reference):")
    print("Prediction counts:")
    print(pd.Series(y_pred).value_counts())
    
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()