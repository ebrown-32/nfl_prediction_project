import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import nfl_data_py as nfl

# Load and process NFL data
""" years = [2022, 2023, 2024]

play_by_play = pd.DataFrame
play_by_play = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)
play_by_play.to_csv("pbp_data.csv") 

df = pd.read_csv('pbp_data.csv')

# Aggregate to game level
game_data = df.groupby(
    ['passer_player_name', 'game_id', 'posteam', 'defteam']
).agg({
    'yards_gained': 'sum',
    'pass_attempt': 'count',
    'pass_touchdown': 'sum',
    'air_yards': 'mean',
    'yards_after_catch': 'mean',
    'shotgun': 'mean',
    'qb_scramble': 'sum'
}).reset_index()

game_data.to_csv('game_data.csv', index=False) """

# Load aggregated data
df = pd.read_csv('game_data.csv')

# Sort by game_id to ensure chronological order
df = df.sort_values('game_id')

def create_sequence_features(df, qb_name, game_idx, max_history=8):
    """
    Create sequence of individual game features with better handling of missing values
    """
    # Get previous games for this QB
    qb_games = df[df['passer_player_name'] == qb_name]
    previous_games = qb_games[qb_games.index < game_idx].tail(max_history)
    
    # Create sequence of game stats
    sequence = []
    for _, game in previous_games.iterrows():
        game_stats = [
            game['yards_gained'] if not np.isnan(game['yards_gained']) else 0,
            game['pass_touchdown'] if not np.isnan(game['pass_touchdown']) else 0,
            game['pass_attempt'] if not np.isnan(game['pass_attempt']) else 0,
            game['air_yards'] if not np.isnan(game['air_yards']) else 0,
            game['yards_after_catch'] if not np.isnan(game['yards_after_catch']) else 0,
            game['shotgun'] if not np.isnan(game['shotgun']) else 0,
            game['qb_scramble'] if not np.isnan(game['qb_scramble']) else 0
        ]
        sequence.append(game_stats)
    
    # Convert to numpy array
    sequence = np.array(sequence)
    
    # Add recency weights if sequence is not empty
    if len(sequence) > 0:
        weights = np.exp(np.linspace(-2, 0, len(sequence)))
        sequence = sequence * weights[:, np.newaxis]
    
    # Pad sequence if needed
    if len(sequence) < max_history:
        padding = np.zeros((max_history - len(sequence), 7))
        sequence = np.vstack([padding, sequence]) if len(sequence) > 0 else padding
    
    return sequence

def create_defense_sequence(df, def_team, game_idx, max_history=4):
    """
    Create sequence of defensive performances with better handling of missing values
    """
    def_games = df[df['defteam'] == def_team]
    previous_games = def_games[def_games.index < game_idx].tail(max_history)
    
    sequence = []
    for _, game in previous_games.iterrows():
        game_stats = [
            game['yards_gained'] if not np.isnan(game['yards_gained']) else 0,
            game['pass_touchdown'] if not np.isnan(game['pass_touchdown']) else 0,
            game['pass_attempt'] if not np.isnan(game['pass_attempt']) else 0,
            game['air_yards'] if not np.isnan(game['air_yards']) else 0,
            game['yards_after_catch'] if not np.isnan(game['yards_after_catch']) else 0
        ]
        sequence.append(game_stats)
    
    # Convert to numpy array
    sequence = np.array(sequence)
    
    # Add recency weights if sequence is not empty
    if len(sequence) > 0:
        weights = np.exp(np.linspace(-1.5, 0, len(sequence)))
        sequence = sequence * weights[:, np.newaxis]
    
    # Pad sequence if needed
    if len(sequence) < max_history:
        padding = np.zeros((max_history - len(sequence), 5))
        sequence = np.vstack([padding, sequence]) if len(sequence) > 0 else padding
    
    return sequence

# Create training data
X_qb_sequences = []
X_def_sequences = []
y_values = []

print("Creating sequences...")
for idx, row in df.iterrows():
    qb_seq = create_sequence_features(df, row['passer_player_name'], idx)
    def_seq = create_defense_sequence(df, row['defteam'], idx)
    
    X_qb_sequences.append(qb_seq)
    X_def_sequences.append(def_seq)
    y_values.append([
        row['yards_gained'],
        row['pass_touchdown'],
        row['pass_attempt']
    ])

# Convert to arrays
X_qb = np.array(X_qb_sequences)
X_def = np.array(X_def_sequences)
y = np.array(y_values)

# Scale the targets only (sequences will be normalized in the model)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# Split the data
from sklearn.model_selection import train_test_split
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# Create QB and team mappings
qb_to_idx = {qb: idx for idx, qb in enumerate(df['passer_player_name'].unique())}
team_to_idx = {team: idx for idx, team in enumerate(df['defteam'].unique())}

class QBDataset(Dataset):
    def __init__(self, X_qb, X_def, y, qb_names, def_teams, indices):
        self.X_qb = torch.FloatTensor(X_qb[indices])
        self.X_def = torch.FloatTensor(X_def[indices])
        self.y = torch.FloatTensor(y[indices])
        self.qb_idx = torch.LongTensor([qb_to_idx[qb] for qb in qb_names[indices]])
        self.team_idx = torch.LongTensor([team_to_idx[team] for team in def_teams[indices]])
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.X_qb[idx],
            self.X_def[idx],
            self.qb_idx[idx],
            self.team_idx[idx],
            self.y[idx]
        )

class QBPredictor(nn.Module):
    def __init__(self, num_qbs, num_teams):
        super().__init__()
        
        # Embeddings for QBs and teams
        self.qb_embedding = nn.Embedding(num_qbs, 16)  # 16-dimensional QB embedding
        self.team_embedding = nn.Embedding(num_teams, 8)  # 8-dimensional team embedding
        
        # Layer Normalization for input sequences
        self.qb_norm = nn.LayerNorm([8, 7])
        self.def_norm = nn.LayerNorm([4, 5])
        
        # GRU layers
        self.qb_gru = nn.GRU(
            input_size=7,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.def_gru = nn.GRU(
            input_size=5,
            hidden_size=16,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Fully connected layers with additional embedding inputs
        self.fc = nn.Sequential(
            nn.Linear(48 + 16 + 8, 64),  # Added dimensions for QB and team embeddings
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 3)
        )
    
    def forward(self, qb_seq, def_seq, qb_idx, team_idx):
        # Get embeddings
        qb_emb = self.qb_embedding(qb_idx)
        team_emb = self.team_embedding(team_idx)
        
        # Process sequences
        qb_seq = self.qb_norm(qb_seq)
        def_seq = self.def_norm(def_seq)
        
        # Process through GRU
        _, qb_hidden = self.qb_gru(qb_seq)
        _, def_hidden = self.def_gru(def_seq)
        
        # Combine all features
        combined = torch.cat([
            qb_hidden[-1],
            def_hidden[-1],
            qb_emb,
            team_emb
        ], dim=1)
        
        return self.fc(combined)

# Add this before creating the DataLoader
print("\nChecking data for NaN values...")
print(f"X_qb NaN count: {np.isnan(X_qb).sum()}")
print(f"X_def NaN count: {np.isnan(X_def).sum()}")
print(f"y NaN count: {np.isnan(y).sum()}")

# If there are NaN values, we should clean them:
X_qb = np.nan_to_num(X_qb, 0)
X_def = np.nan_to_num(X_def, 0)
y = np.nan_to_num(y, 0)

# Create data loaders with smaller batch size
train_dataset = QBDataset(X_qb, X_def, y_scaled, df['passer_player_name'], df['defteam'], train_idx)
test_dataset = QBDataset(X_qb, X_def, y_scaled, df['passer_player_name'], df['defteam'], test_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize model, loss function, and optimizer
print("Initializing model...")
model = QBPredictor(
    num_qbs=len(qb_to_idx),
    num_teams=len(team_to_idx)
)

# Initialize weights properly
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)

model.apply(init_weights)

# Custom loss function to handle potential NaN values
def stable_mse_loss(pred, target):
    mask = ~torch.isnan(target)
    if not torch.any(mask):
        return torch.tensor(0.0, requires_grad=True)
    return torch.mean((pred[mask] - target[mask]) ** 2)

criterion = stable_mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Training loop with better monitoring
print("Starting training...")
num_epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    skipped_batches = 0
    
    for qb_seq, def_seq, qb_idx, team_idx, y_batch in train_loader:
        # Check for invalid values
        if torch.isnan(qb_seq).any() or torch.isnan(def_seq).any():
            skipped_batches += 1
            continue
            
        # Forward pass
        y_pred = model(qb_seq, def_seq, qb_idx, team_idx)
        loss = criterion(y_pred, y_batch)
        
        # Skip bad batches
        if torch.isnan(loss):
            skipped_batches += 1
            continue
            
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    # Validation
    model.eval()
    val_loss = 0
    val_batch_count = 0
    val_skipped = 0
    
    with torch.no_grad():
        for qb_seq, def_seq, qb_idx, team_idx, y_batch in test_loader:
            if torch.isnan(qb_seq).any() or torch.isnan(def_seq).any():
                val_skipped += 1
                continue
                
            val_pred = model(qb_seq, def_seq, qb_idx, team_idx)
            batch_loss = criterion(val_pred, y_batch)
            
            if not torch.isnan(batch_loss):
                val_loss += batch_loss.item()
                val_batch_count += 1
            else:
                val_skipped += 1
    
    # Calculate average losses
    if batch_count > 0:
        avg_train_loss = total_loss / batch_count
    else:
        avg_train_loss = float('nan')
        print(f"Warning: All training batches were skipped in epoch {epoch+1}")
    
    if val_batch_count > 0:
        avg_val_loss = val_loss / val_batch_count
    else:
        avg_val_loss = float('nan')
        print(f"Warning: All validation batches were skipped in epoch {epoch+1}")
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f} (processed {batch_count}/{len(train_loader)} batches)')
        print(f'Validation Loss: {avg_val_loss:.4f} (processed {val_batch_count}/{len(test_loader)} batches)')
        if skipped_batches > 0:
            print(f'Skipped {skipped_batches} training batches due to NaN values')
        if val_skipped > 0:
            print(f'Skipped {val_skipped} validation batches due to NaN values')
        print()
    
    # Early stopping
    if not np.isnan(avg_val_loss):
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def predict_qb_performance(qb_name, opponent_team):
    """Make predictions for a QB against a specific opponent"""
    # Get sequences
    qb_seq = create_sequence_features(df, qb_name, len(df))
    def_seq = create_defense_sequence(df, opponent_team, len(df))
    
    # Convert to tensors and add batch dimension
    qb_seq = torch.FloatTensor(qb_seq).unsqueeze(0)
    def_seq = torch.FloatTensor(def_seq).unsqueeze(0)
    
    # Get QB and team indices
    qb_idx = torch.LongTensor([qb_to_idx[qb_name]])
    team_idx = torch.LongTensor([team_to_idx[opponent_team]])
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        scaled_pred = model(qb_seq, def_seq, qb_idx, team_idx)
        predictions = y_scaler.inverse_transform(scaled_pred)
    
    return {
        'yards': round(predictions[0][0], 1),
        'touchdowns': round(predictions[0][1], 2),
        'attempts': round(predictions[0][2], 1)
    }

# Example predictions
print("\nExample Predictions:")
print("Trevor Lawrence vs Chiefs:")
print(predict_qb_performance("T.Lawrence", "KC"))
print("\nPatrick Mahomes vs Bills:")
print(predict_qb_performance("P.Mahomes", "BUF"))
print("\nPatrick Mahomes vs TEN:")
print(predict_qb_performance("P.Mahomes", "TEN"))
print("\nPatrick Mahomes vs JAX:")
print(predict_qb_performance("P.Mahomes", "JAX"))
print("\nPatrick Mahomes vs BAL:")
print(predict_qb_performance("P.Mahomes", "BAL"))
print("\nPatrick Mahomes vs LV:")
print(predict_qb_performance("P.Mahomes", "LV"))
print("\nJ.Burrow vs JAX:")
print(predict_qb_performance("J.Burrow", "JAX"))
print("\nJ.Burrow vs BAL:")
print(predict_qb_performance("J.Burrow", "BAL"))
print("\nJ.Burrow vs DEN:")
print(predict_qb_performance("J.Burrow", "DEN"))