import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import nfl_data_py as nfl
from tabulate import tabulate
import datetime
import matplotlib.pyplot as plt
from datetime import datetime
import os

"""
Author: ebrown
Date: 11/20/2024
Purpose: Predict QB performance in an upcoming (hypothetical) NFL game based on recent form and opponent defense
Contributors: shout out to Claude Sonnet 3.5 for assistance with the architecture implementation
"""

# Load and process NFL data
years = [2022, 2023, 2024]

print("Loading play by play data...")
play_by_play = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)

# Filter for passing plays
pass_plays = play_by_play[play_by_play['pass_attempt'] == 1].copy()
pass_plays = pass_plays.sort_values(['game_id', 'play_id'])

def create_sequence_features(play_by_play, qb_name, game_id):
    """Creates a sequence of play-by-play data for a quarterback's recent games"""
    qb_plays = play_by_play[
        (play_by_play['passer_player_name'] == qb_name) & 
        (play_by_play['pass_attempt'] == 1)
    ]
    
    # Get plays up to but not including the current game
    previous_plays = qb_plays[qb_plays['game_id'] < game_id]
    
    sequence = []
    for _, play in previous_plays.iterrows():
        play_stats = [
            play['yards_gained'] if not np.isnan(play['yards_gained']) else 0,
            play['pass_touchdown'] if not np.isnan(play['pass_touchdown']) else 0,
            play['complete_pass'] if not np.isnan(play['complete_pass']) else 0,
            play['air_yards'] if not np.isnan(play['air_yards']) else 0,
            play['yards_after_catch'] if not np.isnan(play['yards_after_catch']) else 0,
            play['qb_hit'] if not np.isnan(play['qb_hit']) else 0,
            play['sack'] if not np.isnan(play['sack']) else 0
        ]
        sequence.append(play_stats)
    
    sequence = np.array(sequence)
    
    # Apply linear weighting to emphasize recent plays
    if len(sequence) > 0:
        weights = np.linspace(1.0, 1.5, len(sequence))
        sequence = sequence * weights[:, np.newaxis]
    
    return sequence

def create_defense_sequence(play_by_play, def_team, game_id):
    """Creates a sequence of play-by-play data for a defense's recent games"""
    def_plays = play_by_play[
        (play_by_play['defteam'] == def_team) & 
        (play_by_play['pass_attempt'] == 1)
    ]
    
    # Get plays up to but not including the current game
    previous_plays = def_plays[def_plays['game_id'] < game_id]
    
    sequence = []
    for _, play in previous_plays.iterrows():
        play_stats = [
            play['yards_gained'] if not np.isnan(play['yards_gained']) else 0,
            play['pass_touchdown'] if not np.isnan(play['pass_touchdown']) else 0,
            play['complete_pass'] if not np.isnan(play['complete_pass']) else 0,
            play['air_yards'] if not np.isnan(play['air_yards']) else 0,
            play['yards_after_catch'] if not np.isnan(play['yards_after_catch']) else 0,
            play['qb_hit'] if not np.isnan(play['qb_hit']) else 0,
            play['sack'] if not np.isnan(play['sack']) else 0
        ]
        sequence.append(play_stats)
    
    sequence = np.array(sequence)
    
    # Apply linear weighting to emphasize recent plays
    if len(sequence) > 0:
        weights = np.linspace(1.0, 1.5, len(sequence))
        sequence = sequence * weights[:, np.newaxis]
    
    return sequence

class QBPerformancePredictor(nn.Module):
    def __init__(self, num_qbs, num_teams):
        super().__init__()
        
        self.qb_feature_dim = 7  # Updated for play-level features
        self.def_feature_dim = 7  # Updated for play-level features
        self.hidden_dim = 128
        
        # Add batch normalization
        self.qb_norm = nn.BatchNorm1d(self.qb_feature_dim)
        self.def_norm = nn.BatchNorm1d(self.def_feature_dim)
        
        # Bidirectional LSTM layers
        self.qb_lstm = nn.LSTM(
            input_size=self.qb_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.def_lstm = nn.LSTM(
            input_size=self.def_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Identity embeddings
        self.qb_embedding = nn.Embedding(num_qbs, 128)
        self.team_embedding = nn.Embedding(num_teams, 64)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(self.hidden_dim * 2, num_heads=8, batch_first=True)
        
        # Fully connected layers
        combined_dim = (4 * self.hidden_dim * 2) + 128 + 64
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 5)  # Predicting 5 game-level metrics
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, qb_seq, def_seq, qb_idx, team_idx):
        # Apply batch normalization
        qb_seq = self.qb_norm(qb_seq.transpose(1, 2)).transpose(1, 2)
        def_seq = self.def_norm(def_seq.transpose(1, 2)).transpose(1, 2)
        
        # Process sequences
        qb_out, _ = self.qb_lstm(qb_seq)
        def_out, _ = self.def_lstm(def_seq)
        
        # Attention mechanism
        qb_att, _ = self.attention(qb_out, qb_out, qb_out)
        def_att, _ = self.attention(def_out, def_out, def_out)
        
        # Global average pooling and max pooling
        qb_avg = torch.mean(qb_att, dim=1)
        qb_max = torch.max(qb_att, dim=1)[0]
        def_avg = torch.mean(def_att, dim=1)
        def_max = torch.max(def_att, dim=1)[0]
        
        # Get embeddings
        qb_emb = self.qb_embedding(qb_idx)
        team_emb = self.team_embedding(team_idx)
        
        # Combine features
        combined = torch.cat([qb_avg, qb_max, def_avg, def_max, qb_emb, team_emb], dim=1)
        
        # Forward through FC layers
        x1 = self.relu(self.fc1(combined))
        x1 = self.dropout(x1)
        
        x2 = self.relu(self.fc2(x1))
        x2 = self.dropout(x2)
        
        x3 = self.relu(self.fc3(x2))
        x3 = self.dropout(x3)
        
        out = self.fc4(x3)
        
        return out

class NFLDataset(Dataset):
    def __init__(self, qb_sequences, def_sequences, y, qb_names, def_teams, indices):
        self.qb_seqs = [torch.FloatTensor(qb_sequences[i]) for i in indices]
        self.def_seqs = [torch.FloatTensor(def_sequences[i]) for i in indices]
        self.y = torch.FloatTensor(y[indices])
        self.qb_idx = torch.LongTensor([qb_to_idx[qb] for qb in qb_names[indices]])
        self.team_idx = torch.LongTensor([team_to_idx[team] for team in def_teams[indices]])
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.qb_seqs[idx],
            self.def_seqs[idx],
            self.qb_idx[idx],
            self.team_idx[idx],
            self.y[idx]
        )

# Create sequences and prepare data
print("\nCreating sequences...")
qb_sequences = []
def_sequences = []
y_data = []

# Group plays by game for target creation
game_stats = pass_plays.groupby(['game_id', 'passer_player_name', 'defteam']).agg({
    'yards_gained': 'sum',
    'pass_touchdown': 'sum',
    'interception': 'sum',
    'complete_pass': 'sum',
    'pass_attempt': 'sum',
    'sack': 'sum'
}).reset_index()

game_stats['completion_percentage'] = (game_stats['complete_pass'] / game_stats['pass_attempt'] * 100).round(1)

for _, game in game_stats.iterrows():
    # Create sequences
    qb_seq = create_sequence_features(pass_plays, game['passer_player_name'], game['game_id'])
    def_seq = create_defense_sequence(pass_plays, game['defteam'], game['game_id'])
    
    # Skip if no historical data
    if len(qb_seq) == 0 or len(def_seq) == 0:
        continue
    
    # Create target variables
    target = [
        game['yards_gained'],
        game['pass_touchdown'],
        game['interception'],
        game['completion_percentage'],
        game['sack']
    ]
    
    qb_sequences.append(qb_seq)
    def_sequences.append(def_seq)
    y_data.append(target)

# Create QB and team indices
print("\nCreating indices...")
qb_to_idx = {qb: idx for idx, qb in enumerate(game_stats['passer_player_name'].unique())}
team_to_idx = {team: idx for idx, team in enumerate(game_stats['defteam'].unique())}

# Scale target variables
scaler = StandardScaler()
y = np.array(y_data)
y_scaled = scaler.fit_transform(y)

# Split into train and test sets
print("\nSplitting data...")
train_size = int(0.8 * len(y_scaled))
indices = np.arange(len(y_scaled))
np.random.shuffle(indices)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

def pad_sequences(sequences, max_len=None):
    """Pad sequences to the same length"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_seqs = []
    for seq in sequences:
        if len(seq) == 0:
            padded_seq = np.zeros((max_len, seq.shape[1] if len(seq.shape) > 1 else 1))
        else:
            pad_length = max_len - len(seq)
            if pad_length > 0:
                padding = np.zeros((pad_length, seq.shape[1]))
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq[:max_len]
        padded_seqs.append(padded_seq)
    
    return np.array(padded_seqs)

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    qb_seqs, def_seqs, qb_idx, team_idx, y = zip(*batch)
    
    # Pad sequences
    qb_seqs_padded = pad_sequences([seq.numpy() for seq in qb_seqs])
    def_seqs_padded = pad_sequences([seq.numpy() for seq in def_seqs])
    
    return (
        torch.FloatTensor(qb_seqs_padded),
        torch.FloatTensor(def_seqs_padded),
        torch.stack(qb_idx),
        torch.stack(team_idx),
        torch.stack(y)
    )

# Create data loaders
train_dataset = NFLDataset(qb_sequences, def_sequences, y_scaled, 
                          game_stats['passer_player_name'].values, 
                          game_stats['defteam'].values, train_idx)
test_dataset = NFLDataset(qb_sequences, def_sequences, y_scaled, 
                         game_stats['passer_player_name'].values, 
                         game_stats['defteam'].values, test_idx)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

# Initialize model and training components
print("\nInitializing model...")
model = QBPerformancePredictor(
    num_qbs=len(qb_to_idx),
    num_teams=len(team_to_idx)
)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training loop
print("\nStarting training...")
num_epochs = 100
best_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        qb_seq, def_seq, qb_idx, team_idx, y_batch = batch
        
        optimizer.zero_grad()
        y_pred = model(qb_seq, def_seq, qb_idx, team_idx)
        loss = criterion(y_pred, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            qb_seq, def_seq, qb_idx, team_idx, y_batch = batch
            y_pred = model(qb_seq, def_seq, qb_idx, team_idx)
            val_loss += criterion(y_pred, y_batch).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}\n')
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

print("Training completed!")

def predict_qb_performance(qb_name, def_team):
    """Make predictions for a QB against a specific defense"""
    # Ensure the QB and team exist in the data
    if qb_name not in qb_to_idx:
        raise ValueError(f"Quarterback {qb_name} not found in data.")
    if def_team not in team_to_idx:
        raise ValueError(f"Defense team {def_team} not found in data.")
    
    # Create sequences using all available historical data
    qb_seq = create_sequence_features(pass_plays, qb_name, float('inf'))
    def_seq = create_defense_sequence(pass_plays, def_team, float('inf'))
    
    if len(qb_seq) == 0:
        raise ValueError(f"No historical data found for QB: {qb_name}")
    if len(def_seq) == 0:
        raise ValueError(f"No historical data found for defense: {def_team}")
    
    # Convert to tensors
    qb_seq_tensor = torch.FloatTensor(qb_seq).unsqueeze(0)
    def_seq_tensor = torch.FloatTensor(def_seq).unsqueeze(0)
    qb_idx = torch.LongTensor([qb_to_idx[qb_name]])
    team_idx = torch.LongTensor([team_to_idx[def_team]])
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(qb_seq_tensor, def_seq_tensor, qb_idx, team_idx)
    
    # Inverse transform the prediction
    prediction = scaler.inverse_transform(prediction.numpy())
    
    # Round predictions to reasonable values
    return {
        'yards_gained': round(float(prediction[0, 0]), 1),
        'pass_touchdown': round(float(prediction[0, 1]), 1),
        'interception': round(float(prediction[0, 2]), 1),
        'completion_percentage': round(float(prediction[0, 3]), 1),
        'sack': round(float(prediction[0, 4]), 1)
    }

# Example usage
if __name__ == "__main__":
    try:
        prediction = predict_qb_performance("P.Mahomes", "BUF")
        print("\nPredicted QB Performance:")
        for stat, value in prediction.items():
            print(f"{stat}: {value}")
    except ValueError as e:
        print(e)


