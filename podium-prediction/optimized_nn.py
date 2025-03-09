import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import json
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define our target variable based on position
def create_target(position):
    # Convert position to string to handle different data types
    position_str = str(position).strip()
    if position_str == "1" or position_str == "1.0":
        return 'first'
    elif position_str == "2" or position_str == "2.0":
        return 'second'
    elif position_str == "3" or position_str == "3.0":
        return 'third'
    else:
        return 'off_podium'


def apply_labels(df):
    df['label'] = df['position'].apply(create_target)
    # Drop the position column
    df = df.drop('position', axis=1)
    return df


class PodiumNN(nn.Module):
    """
    PyTorch Neural Network for F1 podium prediction
    """
    def __init__(self, input_dim, hidden_dims, dropout_rates, num_classes):
        super(PodiumNN, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rates[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i+1]))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=10):
    """
    Train the PyTorch model with early stopping and improved numerical stability
    """
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training history
    train_losses = []
    val_losses = []
    
    print("\nTraining Neural Network model...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Check for NaN or Inf values in inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value: {loss.item()}. Skipping batch.")
                continue
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else float('inf')
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Check for NaN or Inf values in inputs
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Check if loss is valid
                if not torch.isnan(loss) and not torch.isinf(loss):
                    running_val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss = running_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('inf')
        val_acc = correct / total if total > 0 else 0
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def check_data_issues(df):
    """Check for common data issues"""
    print("\nData Diagnostics:")
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    print(f"Total NaN values: {nan_count}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    print(f"Total infinite values: {inf_count}")
    
    # Check position column
    if 'position' in df.columns:
        print(f"Position column data type: {df['position'].dtype}")
        print(f"Position column unique values: {df['position'].unique()[:10]} (showing first 10)")
    
    return nan_count, inf_count


def main():
    # Check if a grid search results file is provided as an argument
    if len(sys.argv) > 1:
        grid_search_file = sys.argv[1]
    else:
        # Find the most recent grid search results file
        grid_search_files = [f for f in os.listdir('.') if f.startswith('grid_search_results_') and f.endswith('.json')]
        if not grid_search_files:
            print("No grid search results found. Please run grid_search.py first or provide a results file.")
            return
        grid_search_file = sorted(grid_search_files)[-1]  # Get the most recent file
    
    print(f"Using grid search results from: {grid_search_file}")
    
    # Load the best parameters from the grid search results
    with open(grid_search_file, 'r') as f:
        grid_search_results = json.load(f)
    
    best_params = grid_search_results['best_params']
    print(f"Best parameters: {best_params}")
    
    # Load the data from the new CSV file
    df = pd.read_csv("../cleaned/f1_podium_prediction_dataset.csv")
    
    # Check for data issues
    nan_count, inf_count = check_data_issues(df)
    
    # Clean data if needed
    if nan_count > 0 or inf_count > 0:
        print("Cleaning data issues...")
        # Fill NaN values with 0
        df = df.fillna(0)
        
        # Replace infinite values with large but finite numbers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], [1e6, -1e6])

    # Apply labels to the data
    df = apply_labels(df)
    
    # Drop columns that shouldn't be used for training
    columns_to_drop = ['podium', 'driver_name', 'constructor_name', 'points']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

    # Split data into training (before 2024) and testing (2024)
    train_df = df[(df['year'] < 2024) & (df['year'] >= 2000)]
    test_df = df[df['year'] == 2024]

    # Define categorical columns for one-hot encoding
    categorical_cols = ['driverId', 'constructorId', 'circuitId']
    
    # Define numerical columns (all columns except categorical and label)
    numerical_cols = [col for col in train_df.columns if col not in categorical_cols + ['label']]
    
    # Print column information
    print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols[:5]}...")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
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
    
    # Fit the preprocessor on training data
    preprocessor.fit(X_train)
    
    # Transform the data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Check for NaN or infinite values in processed data
    if hasattr(X_train_processed, 'toarray'):
        X_train_array = X_train_processed.toarray()
        X_test_array = X_test_processed.toarray()
    else:
        X_train_array = X_train_processed
        X_test_array = X_test_processed
    
    print(f"\nProcessed training data shape: {X_train_array.shape}")
    print(f"NaN values in training data: {np.isnan(X_train_array).any()}")
    print(f"Infinite values in training data: {np.isinf(X_train_array).any()}")
    
    # Replace any NaN or infinite values
    X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=1.0, neginf=-1.0)
    X_test_array = np.nan_to_num(X_test_array, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Encode string labels to numeric values
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Print label distribution
    print(f"\nLabel distribution in training data:")
    for i, label in enumerate(label_encoder.classes_):
        count = (y_train_encoded == i).sum()
        print(f"  {label}: {count} ({count/len(y_train_encoded):.2%})")
    
    # Store the mapping for later reference
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print("Label mapping:", label_mapping)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_array)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_test_tensor = torch.FloatTensor(X_test_array)
    y_test_tensor = torch.LongTensor(y_test_encoded)
    
    # Split training data into training and validation sets
    val_size = int(0.2 * len(X_train_tensor))
    train_size = len(X_train_tensor) - val_size
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor[:train_size], y_train_tensor[:train_size])
    val_dataset = TensorDataset(X_train_tensor[train_size:], y_train_tensor[train_size:])
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders with the best batch size
    batch_size = best_params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get the input dimension after preprocessing
    input_dim = X_train_tensor.shape[1]
    num_classes = len(label_encoder.classes_)
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    
    # Initialize the model with the best parameters
    model = PodiumNN(
        input_dim=input_dim,
        hidden_dims=best_params['hidden_dims'],
        dropout_rates=best_params['dropout_rates'],
        num_classes=num_classes
    ).to(device)
    
    # Define loss function and optimizer with the best learning rate and weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # Train the model
    try:
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, 
            epochs=50, patience=10
        )
        
        # Evaluate the model on test data
        model.eval()
        correct = 0
        total = 0
        
        # Get prediction probabilities
        y_proba = np.zeros((len(X_test_tensor), num_classes))
        
        print("\nEvaluating model on test data...")
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Check for NaN or Inf values in inputs
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Convert to probabilities using softmax
                probs = nn.functional.softmax(outputs, dim=1)
                
                # Store probabilities
                start_idx = i * test_loader.batch_size
                end_idx = min((i + 1) * test_loader.batch_size, len(X_test_tensor))
                y_proba[start_idx:end_idx] = probs.cpu().numpy()
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        test_acc = correct / total if total > 0 else 0
        print(f"Test accuracy: {test_acc:.4f}")
        
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
        
        print("\nPredicted vs Actual Podiums for Each Race (Optimized PyTorch Neural Network):")
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
        print("\nCustom F1 Podium Prediction Metrics (Optimized PyTorch Neural Network):")
        
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
        
        # Standard evaluation metrics on the original predictions
        y_pred_encoded = np.argmax(y_proba, axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        print("\n" + "=" * 80)
        print("\nStandard Classification Metrics (for reference):")
        print("Prediction counts:")
        print(pd.Series(y_pred).value_counts())
        
        print("\nAccuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_dims': best_params['hidden_dims'],
            'dropout_rates': best_params['dropout_rates'],
            'num_classes': num_classes,
            'label_mapping': label_mapping
        }, 'optimized_podium_nn_model.pth')
        
        print("\nModel saved to 'optimized_podium_nn_model.pth'")
    
    except Exception as e:
        print(f"Error during model training or evaluation: {str(e)}")
        print("Please check the data and parameters and try again.")


if __name__ == "__main__":
    main() 