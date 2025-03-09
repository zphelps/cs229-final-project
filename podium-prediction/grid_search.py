import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import itertools
import time
import json
from datetime import datetime
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


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, device, epochs=30, patience=7):
    """
    Train the model and evaluate on validation set with improved numerical stability
    """
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Check for NaN or Inf values in inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Warning: NaN or Inf values detected in inputs. Cleaning data...")
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else float('inf')
        
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
        
        # Print epoch results
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return val_loss, val_acc


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
    
    # Check for extreme values
    for col in numeric_cols:
        if df[col].min() < -1e10 or df[col].max() > 1e10:
            print(f"Column {col} has extreme values: min={df[col].min()}, max={df[col].max()}")
    
    return nan_count, inf_count


def grid_search():
    print("Starting grid search for hyperparameter optimization...")
    start_time = time.time()
    
    # Load the data
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
    
    # Apply labels
    df = apply_labels(df)
    
    # Drop columns that shouldn't be used for training
    columns_to_drop = ['podium', 'driver_name', 'constructor_name', 'points']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
    
    # Split data into training and validation
    train_df = df[(df['year'] < 2023) & (df['year'] >= 2000)]
    val_df = df[df['year'] == 2023]  # Use 2023 as validation
    
    # Define categorical columns for one-hot encoding
    categorical_cols = ['driverId', 'constructorId', 'circuitId']
    
    # Define numerical columns
    numerical_cols = [col for col in train_df.columns if col not in categorical_cols + ['label']]
    
    # Print column information
    print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols[:5]}...")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Apply preprocessing
    X_train = train_df.drop(['label'], axis=1)
    y_train = train_df['label']
    X_val = val_df.drop(['label'], axis=1)
    y_val = val_df['label']
    
    # Fit the preprocessor on training data
    preprocessor.fit(X_train)
    
    # Transform the data
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Check for NaN or infinite values in processed data
    if hasattr(X_train_processed, 'toarray'):
        X_train_array = X_train_processed.toarray()
        X_val_array = X_val_processed.toarray()
    else:
        X_train_array = X_train_processed
        X_val_array = X_val_processed
    
    print(f"\nProcessed training data shape: {X_train_array.shape}")
    print(f"NaN values in training data: {np.isnan(X_train_array).any()}")
    print(f"Infinite values in training data: {np.isinf(X_train_array).any()}")
    
    # Replace any NaN or infinite values
    X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=1.0, neginf=-1.0)
    X_val_array = np.nan_to_num(X_val_array, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # Print label distribution
    print(f"\nLabel distribution in training data:")
    for i, label in enumerate(label_encoder.classes_):
        count = (y_train_encoded == i).sum()
        print(f"  {label}: {count} ({count/len(y_train_encoded):.2%})")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_array)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_val_tensor = torch.FloatTensor(X_val_array)
    y_val_tensor = torch.LongTensor(y_val_encoded)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define a smaller hyperparameter grid for initial testing
    param_grid = {
        'hidden_dims': [
            [64, 32],
            [128, 64]
        ],
        'dropout_rates': [
            [0.2, 0.1],
            [0.3, 0.2]
        ],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [32, 64],
        'weight_decay': [0, 1e-4]
    }
    
    # Ensure hidden_dims and dropout_rates have matching lengths
    valid_combinations = []
    for hidden_dim in param_grid['hidden_dims']:
        for dropout_rate in param_grid['dropout_rates']:
            if len(hidden_dim) == len(dropout_rate):
                valid_combinations.append((hidden_dim, dropout_rate))
    
    # Generate all valid parameter combinations
    all_combinations = []
    for hidden_dim, dropout_rate in valid_combinations:
        for lr in param_grid['learning_rate']:
            for bs in param_grid['batch_size']:
                for wd in param_grid['weight_decay']:
                    all_combinations.append({
                        'hidden_dims': hidden_dim,
                        'dropout_rates': dropout_rate,
                        'learning_rate': lr,
                        'batch_size': bs,
                        'weight_decay': wd
                    })
    
    print(f"Total parameter combinations to try: {len(all_combinations)}")
    
    # Track results
    results = []
    best_val_loss = float('inf')
    best_params = None
    
    # Input dimension
    input_dim = X_train_tensor.shape[1]
    num_classes = len(label_encoder.classes_)
    print(f"Input dimension: {input_dim}, Number of classes: {num_classes}")
    
    # Grid search
    for i, params in enumerate(all_combinations):
        print(f"\nTrying combination {i+1}/{len(all_combinations)}: {params}")
        
        # Create data loaders with current batch size
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Create model with current parameters
        model = PodiumNN(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            dropout_rates=params['dropout_rates'],
            num_classes=num_classes
        ).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        try:
            # Train and evaluate
            val_loss, val_acc = train_and_evaluate(
                model, train_loader, val_loader, criterion, optimizer, device
            )
            
            # Record results
            result = {
                'params': params,
                'val_loss': float(val_loss),  # Convert to Python float for JSON serialization
                'val_accuracy': float(val_acc)
            }
            results.append(result)
            
            print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
            
            # Update best parameters
            if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
                best_val_loss = val_loss
                best_params = params
                print(f"New best parameters found!")
        
        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Record failure
            results.append({
                'params': params,
                'val_loss': float('inf'),
                'val_accuracy': 0.0,
                'error': str(e)
            })
    
    # Sort results by validation loss
    valid_results = [r for r in results if not np.isnan(r['val_loss']) and not np.isinf(r['val_loss'])]
    
    if valid_results:
        valid_results.sort(key=lambda x: x['val_loss'])
        best_result = valid_results[0]
        best_params = best_result['params']
        best_val_loss = best_result['val_loss']
    else:
        print("No valid results found. Using default parameters.")
        best_params = {
            'hidden_dims': [64, 32],
            'dropout_rates': [0.2, 0.1],
            'learning_rate': 0.0001,
            'batch_size': 32,
            'weight_decay': 1e-4
        }
        best_val_loss = float('inf')
    
    # Print best parameters
    print("\n" + "=" * 50)
    print("Grid Search Complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"grid_search_results_{timestamp}.json", "w") as f:
        json.dump({
            'best_params': best_params,
            'best_val_loss': best_val_loss,
            'all_results': results
        }, f, indent=4, default=str)  # Use default=str to handle non-serializable objects
    
    print(f"Results saved to grid_search_results_{timestamp}.json")
    
    return best_params


if __name__ == "__main__":
    best_params = grid_search() 