import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------------------------------
# 1. Read and split data
# -----------------------------------------------------
def load_data(csv_path="cleaned/results2.csv"):
    """
    Reads the CSV, splits into training and test sets (year==2024 for test),
    and applies LabelEncoding for categorical fields.
    
    Returns:
        train_df, test_df, encoders (dict of label encoders)
    """
    df = pd.read_csv(csv_path)
    
    # Split into train/test by year
    train_df = df[(df['year'] >= 2000) & (df['year'] <= 2023)].copy()
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
    
    # Add StandardScaler for numeric columns
    numeric_cols = ['year', 'round', 'grid', 'driver_points', 'constructor_points', 'driver_wins', 'constructor_wins']
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    train_df['label'] = train_df['position'].isin(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]).astype(float)
    test_df['label'] = test_df['position'].isin(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]).astype(float)
    
    return train_df, test_df, encoders

# -----------------------------------------------------
# 2. Create a custom Dataset
# -----------------------------------------------------
class F1Dataset(Dataset):
    """
    Expects a dataframe with columns including:
      - driverId (encoded integer)
      - constructorId (encoded integer)
      - circuitId (encoded integer)
      - year, round, grid, position, driver_points, constructor_points, ...
    
    The label is derived from position (1-3 => 1, otherwise => 0).
    """
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        
        # Identify categorical columns to embed
        self.driver_ids = torch.tensor(self.df['driverId'].values, dtype=torch.long)
        self.circuit_ids = torch.tensor(self.df['circuitId'].values, dtype=torch.long)
        self.constructor_ids = torch.tensor(self.df['constructorId'].values, dtype=torch.long)
        
        # Updated numeric columns
        num_cols = ['year', 'round', 'grid', 'driver_points', 'constructor_points', 'driver_wins', 'constructor_wins']
        self.numerics = torch.tensor(self.df[num_cols].values, dtype=torch.float)
        
        # Create label from 'position': label = 1 if position in ["1", "2", "3"] else 0
        # self.labels = torch.tensor(self.df['position'].isin(["1", "2", "3"]).astype(float).values, dtype=torch.float)
         # Now label is always from the 'label' column
        if 'label' not in self.df.columns:
            raise ValueError("No 'label' column found in the dataframe.")
        self.labels = torch.tensor(self.df['label'].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        driver_id = self.driver_ids[idx]
        circuit_id = self.circuit_ids[idx]
        constructor_id = self.constructor_ids[idx]
        numeric_features = self.numerics[idx]
        label = self.labels[idx]
        
        return (driver_id, circuit_id, constructor_id, numeric_features), label

# -----------------------------------------------------
# 3. Define the Model
# -----------------------------------------------------
class F1Predictor(nn.Module):
    def __init__(self,
                 num_drivers,   # Cardinality of driverId
                 num_circuits,  # Cardinality of circuitId
                 num_constructors,  # Cardinality of constructorId
                 embedding_dim_driver=8,
                 embedding_dim_circuit=4,
                 embedding_dim_constructor=4,
                 num_numeric=7,  # year, round, grid, driver_points, constructor_points, driver_wins, constructor_wins
                 hidden_dim=32,
                 dropout=0.3):
        super().__init__()
        
        # Embeddings
        self.driver_emb = nn.Embedding(num_drivers, embedding_dim_driver)
        self.circuit_emb = nn.Embedding(num_circuits, embedding_dim_circuit)
        self.constructor_emb = nn.Embedding(num_constructors, embedding_dim_constructor)
        
        # Calculate total embedding dim
        total_emb_dim = embedding_dim_driver + embedding_dim_circuit + embedding_dim_constructor
        
        # A simple feed-forward network
        # self.fc1 = nn.Linear(total_emb_dim + num_numeric, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, 1)
        
        # self.relu = nn.ReLU()

        # Deeper network with dropout
        self.fc1 = nn.Linear(total_emb_dim + num_numeric, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, driver_id, circuit_id, constructor_id, numeric_features):
        # Embed the categorical features
        d_emb = self.driver_emb(driver_id)
        c_emb = self.circuit_emb(circuit_id)
        cons_emb = self.constructor_emb(constructor_id)
        
        # Concatenate embeddings + numeric features
        x = torch.cat([d_emb, c_emb, cons_emb, numeric_features], dim=1)
        
        # # MLP forward
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # # Output layer (logits)
        # logits = self.fc3(x)
        
        # MLP forward with dropout and batch normalization
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        logits = self.fc4(x)
        
        return logits.view(-1)

# -----------------------------------------------------
# 4. Training function
# -----------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    for (driver_id, circuit_id, constructor_id, numeric_feats), labels in dataloader:
        driver_id = driver_id.to(device)
        circuit_id = circuit_id.to(device)
        constructor_id = constructor_id.to(device)
        numeric_feats = numeric_feats.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(driver_id, circuit_id, constructor_id, numeric_feats)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
    
    return total_loss / len(dataloader.dataset)

# -----------------------------------------------------
# 5. Evaluation function
# -----------------------------------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    TP = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for (driver_id, circuit_id, constructor_id, numeric_feats), labels in dataloader:
            driver_id = driver_id.to(device)
            circuit_id = circuit_id.to(device)
            constructor_id = constructor_id.to(device)
            numeric_feats = numeric_feats.to(device)
            labels = labels.to(device)
            
            outputs = model(driver_id, circuit_id, constructor_id, numeric_feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()

            # print("--------------------------------")
            # print(f"{preds.tolist()}\n{labels.tolist()}")

            # Count true positives, false positives, and false negatives
            TP += ((preds == 1) & (labels == 1)).sum().item()
            FP += ((preds == 1) & (labels == 0)).sum().item()
            FN += ((preds == 0) & (labels == 1)).sum().item()
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    
    # Compute precision, recall, and f1
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return avg_loss, accuracy, precision, recall, f1

# -----------------------------------------------------
# 6. Main training script
# -----------------------------------------------------
def main(csv_path='cleaned/results2.csv',
         batch_size=32,
         epochs=50,
         lr=1e-3,
         embedding_dim_driver=8,
         embedding_dim_circuit=4,
         embedding_dim_constructor=4,
         hidden_dim=32):
    
    # Load data
    train_df, test_df, encoders = load_data(csv_path)
    
    # Create datasets
    train_dataset = F1Dataset(train_df)
    test_dataset = F1Dataset(test_df)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get cardinalities
    num_drivers = len(encoders['driverId'].classes_)
    num_circuits = len(encoders['circuitId'].classes_)
    num_constructors = len(encoders['constructorId'].classes_)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = F1Predictor(
        num_drivers=num_drivers,
        num_circuits=num_circuits,
        num_constructors=num_constructors,
        embedding_dim_driver=embedding_dim_driver,
        embedding_dim_circuit=embedding_dim_circuit,
        embedding_dim_constructor=embedding_dim_constructor,
        num_numeric=7,  # We have 7 numeric features
        hidden_dim=hidden_dim
    ).to(device)
    
    # Define optimizer and loss
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Calculate class weights to handle imbalance
    # pos_weight = (train_df['position'].isin(["1", "2", "3"]) == False).sum() / \
    #              (train_df['position'].isin(["1", "2", "3"])).sum()
    
    # # Use weighted loss
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Use learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Training loop
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, test_loader, criterion, device)
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, "
              f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")

    # Once training is done, do a final evaluation on the test set:
    final_loss, final_acc, final_prec, final_rec, final_f1 = evaluate(model, test_loader, criterion, device)
    print("\nFinal Test Evaluation:")
    print(f"Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}, "
          f"Precision: {final_prec:.4f}, Recall: {final_rec:.4f}, F1: {final_f1:.4f}")

    # Optional: Save model
    # torch.save(model.state_dict(), "f1_win_predictor.pt")

if __name__ == "__main__":
    main()
