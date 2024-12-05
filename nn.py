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

# Filter for passing plays and aggregate to game level
pass_plays = play_by_play[play_by_play['pass_attempt'] == 1]

print("\nAggregating game-level statistics...")
df = pass_plays.groupby(['game_id', 'passer_player_name', 'posteam', 'defteam', 'week', 'season']).agg({
    # Basic passing stats
    'pass_attempt': 'sum',
    'complete_pass': 'sum',
    'yards_gained': 'sum',
    'pass_touchdown': 'sum',
    'interception': 'sum',
    
    # Advanced metrics
    'air_yards': 'mean',
    'yards_after_catch': 'mean',
    'qb_hit': 'sum',
    'sack': 'sum'
}).reset_index()

# Calculate derived metrics
df['completion_percentage'] = (df['complete_pass'] / df['pass_attempt'] * 100).round(1)
df['yards_per_attempt'] = (df['yards_gained'] / df['pass_attempt']).round(1)
df['sack_rate'] = (df['sack'] / (df['pass_attempt'] + df['sack']) * 100).round(1)

# Clean up and filter data
df = df.dropna(subset=['passer_player_name', 'pass_attempt'])
df = df[df['pass_attempt'] >= 10]  # Min attempts filter

# Sort by game_id to ensure chronological order
df = df.sort_values('game_id')

def create_play_by_play_sequences(play_by_play, qb_name, game_id, max_plays=100):
    """Create sequence of play-by-play data for a specific game and QB"""
    qb_plays = play_by_play[(play_by_play['passer_player_name'] == qb_name) & 
                           (play_by_play['game_id'] == game_id)]
    qb_plays = qb_plays.head(max_plays)

    sequence = []
    for _, play in qb_plays.iterrows():
        play_stats = [
            play['yards_gained'] if not np.isnan(play['yards_gained']) else 0,
            play['pass_touchdown'] if not np.isnan(play['pass_touchdown']) else 0,
            play['interception'] if not np.isnan(play['interception']) else 0,
            play['air_yards'] if not np.isnan(play['air_yards']) else 0,
            play['yards_after_catch'] if not np.isnan(play['yards_after_catch']) else 0,
            play['qb_hit'] if not np.isnan(play['qb_hit']) else 0,
            play['sack'] if not np.isnan(play['sack']) else 0
        ]
        sequence.append(play_stats)

    sequence = np.array(sequence)
    
    # Pad if necessary
    if len(sequence) < max_plays:
        padding = np.zeros((max_plays - len(sequence), 7))
        sequence = np.vstack([sequence, padding]) if len(sequence) > 0 else padding

    return sequence

def create_sequence_features(df, qb_name, game_idx, max_history=16):
    """Creates a sequence of game statistics for a quarterback's recent performances"""
    qb_games = df[df['passer_player_name'] == qb_name]
    previous_games = qb_games[qb_games.index < game_idx].tail(max_history)
    
    sequence = []
    for _, game in previous_games.iterrows():
        game_stats = [
            game['yards_gained'] if not np.isnan(game['yards_gained']) else 0,
            game['pass_touchdown'] if not np.isnan(game['pass_touchdown']) else 0,
            game['pass_attempt'] if not np.isnan(game['pass_attempt']) else 0,
            game['air_yards'] if not np.isnan(game['air_yards']) else 0,
            game['yards_after_catch'] if not np.isnan(game['yards_after_catch']) else 0,
            game['qb_hit'] if not np.isnan(game['qb_hit']) else 0,
            game['sack'] if not np.isnan(game['sack']) else 0,
            game['completion_percentage'] if not np.isnan(game['completion_percentage']) else 0,
            game['yards_per_attempt'] if not np.isnan(game['yards_per_attempt']) else 0,
            game['sack_rate'] if not np.isnan(game['sack_rate']) else 0
        ]
        sequence.append(game_stats)
    
    sequence = np.array(sequence)
    
    # Apply exponential weighting
    if len(sequence) > 0:
        weights = np.exp(np.linspace(-2, 0, len(sequence)))
        sequence = sequence * weights[:, np.newaxis]
    
    # Pad if necessary
    if len(sequence) < max_history:
        padding = np.zeros((max_history - len(sequence), 10))
        sequence = np.vstack([padding, sequence]) if len(sequence) > 0 else padding
    
    return sequence

def create_defense_sequence(df, def_team, game_idx, max_history=8):
    """Create sequence of defensive performances"""
    def_games = df[df['defteam'] == def_team]
    previous_games = def_games[def_games.index < game_idx].tail(max_history)
    
    sequence = []
    for _, game in previous_games.iterrows():
        game_stats = [
            game['yards_gained'] if not np.isnan(game['yards_gained']) else 0,
            game['pass_touchdown'] if not np.isnan(game['pass_touchdown']) else 0,
            game['pass_attempt'] if not np.isnan(game['pass_attempt']) else 0,
            game['air_yards'] if not np.isnan(game['air_yards']) else 0,
            game['yards_after_catch'] if not np.isnan(game['yards_after_catch']) else 0,
            game['qb_hit'] if not np.isnan(game['qb_hit']) else 0,
            game['sack'] if not np.isnan(game['sack']) else 0,
            game['completion_percentage'] if not np.isnan(game['completion_percentage']) else 0,
            game['sack_rate'] if not np.isnan(game['sack_rate']) else 0
        ]
        sequence.append(game_stats)
    
    sequence = np.array(sequence)
    
    if len(sequence) > 0:
        weights = np.exp(np.linspace(-1.5, 0, len(sequence)))
        sequence = sequence * weights[:, np.newaxis]
    
    if len(sequence) < max_history:
        padding = np.zeros((max_history - len(sequence), 9))
        sequence = np.vstack([padding, sequence]) if len(sequence) > 0 else padding
    
    return sequence

class QBPerformancePredictor(nn.Module):
    def __init__(self, num_qbs, num_teams, qb_seq_length=16, def_seq_length=8, pbp_seq_length=100):
        super().__init__()
        
        self.qb_feature_dim = 10  # Number of features in QB sequence
        self.def_feature_dim = 9  # Number of features in defense sequence
        self.pbp_feature_dim = 7  # Number of features in play-by-play data
        self.hidden_dim = 64
        
        # Layer normalization for input sequences
        self.qb_norm = nn.LayerNorm([qb_seq_length, self.qb_feature_dim])
        self.def_norm = nn.LayerNorm([def_seq_length, self.def_feature_dim])
        self.pbp_norm = nn.LayerNorm([pbp_seq_length, self.pbp_feature_dim])
        
        # LSTM layers for sequence processing
        self.qb_lstm = nn.LSTM(
            input_size=self.qb_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.def_lstm = nn.LSTM(
            input_size=self.def_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.pbp_lstm = nn.LSTM(
            input_size=self.pbp_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Identity embeddings
        self.qb_embedding = nn.Embedding(num_qbs, 64)
        self.team_embedding = nn.Embedding(num_teams, 32)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True)
        
        # Fully connected layers for final prediction
        combined_dim = (3 * self.hidden_dim) + 64 + 32  # 3 LSTM outputs + QB embedding + Team embedding
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)  # 5 output metrics
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, qb_seq, def_seq, pbp_seq, qb_idx, team_idx):
        # Normalize sequences
        qb_seq = self.qb_norm(qb_seq)
        def_seq = self.def_norm(def_seq)
        pbp_seq = self.pbp_norm(pbp_seq)
        
        # Process sequences through LSTMs
        qb_out, _ = self.qb_lstm(qb_seq)
        def_out, _ = self.def_lstm(def_seq)
        pbp_out, _ = self.pbp_lstm(pbp_seq)
        
        # Apply attention to each sequence
        qb_att, _ = self.attention(qb_out, qb_out, qb_out)
        def_att, _ = self.attention(def_out, def_out, def_out)
        pbp_att, _ = self.attention(pbp_out, pbp_out, pbp_out)
        
        # Get final representations
        qb_final = qb_att[:, -1, :]
        def_final = def_att[:, -1, :]
        pbp_final = pbp_att[:, -1, :]
        
        # Get identity embeddings
        qb_emb = self.qb_embedding(qb_idx)
        team_emb = self.team_embedding(team_idx)
        
        # Combine all features
        combined = torch.cat([qb_final, def_final, pbp_final, qb_emb, team_emb], dim=1)
        
        # Final prediction layers
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class NFLDataset(Dataset):
    def __init__(self, X_qb, X_def, X_pbp, y, qb_names, def_teams, indices):
        self.X_qb = torch.FloatTensor(X_qb[indices])
        self.X_def = torch.FloatTensor(X_def[indices])
        self.X_pbp = torch.FloatTensor(X_pbp[indices])
        self.y = torch.FloatTensor(y[indices])
        self.qb_idx = torch.LongTensor([qb_to_idx[qb] for qb in qb_names[indices]])
        self.team_idx = torch.LongTensor([team_to_idx[team] for team in def_teams[indices]])
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.X_qb[idx],
            self.X_def[idx],
            self.X_pbp[idx],
            self.qb_idx[idx],
            self.team_idx[idx],
            self.y[idx]
        )

# Create sequences and prepare data
print("\nCreating sequences...")
X_qb_sequences = []
X_def_sequences = []
X_pbp_sequences = []
y_data = []

for idx, row in df.iterrows():
    # Create sequences
    qb_seq = create_sequence_features(df, row['passer_player_name'], idx)
    def_seq = create_defense_sequence(df, row['defteam'], idx)
    pbp_seq = create_play_by_play_sequences(play_by_play, row['passer_player_name'], row['game_id'])
    
    # Create target variables
    target = [
        row['yards_gained'],
        row['pass_touchdown'],
        row['interception'],
        row['completion_percentage'],
        row['sack']
    ]
    
    X_qb_sequences.append(qb_seq)
    X_def_sequences.append(def_seq)
    X_pbp_sequences.append(pbp_seq)
    y_data.append(target)

# Convert to numpy arrays
X_qb = np.array(X_qb_sequences)
X_def = np.array(X_def_sequences)
X_pbp = np.array(X_pbp_sequences)
y = np.array(y_data)

# Create QB and team indices
print("\nCreating indices...")
qb_to_idx = {qb: idx for idx, qb in enumerate(df['passer_player_name'].unique())}
team_to_idx = {team: idx for idx, team in enumerate(df['defteam'].unique())}

# Scale target variables
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y)

# Split into train and test sets
print("\nSplitting data...")
train_size = int(0.8 * len(df))
indices = np.arange(len(df))
np.random.shuffle(indices)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

# Create data loaders
train_dataset = NFLDataset(X_qb, X_def, X_pbp, y_scaled, df['passer_player_name'].values, df['defteam'].values, train_idx)
test_dataset = NFLDataset(X_qb, X_def, X_pbp, y_scaled, df['passer_player_name'].values, df['defteam'].values, test_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize model and training components
print("\nInitializing model...")
model = QBPerformancePredictor(
    num_qbs=len(qb_to_idx),
    num_teams=len(team_to_idx)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
print("\nStarting training...")
num_epochs = 50
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        qb_seq, def_seq, pbp_seq, qb_idx, team_idx, y_batch = batch
        
        optimizer.zero_grad()
        y_pred = model(qb_seq, def_seq, pbp_seq, qb_idx, team_idx)
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
            qb_seq, def_seq, pbp_seq, qb_idx, team_idx, y_batch = batch
            y_pred = model(qb_seq, def_seq, pbp_seq, qb_idx, team_idx)
            val_loss += criterion(y_pred, y_batch).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(test_loader)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Training Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}\n')
    
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

# Load best model for predictions
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

def predict_qb_performance(qb_name, def_team):
    # Ensure the QB and team exist in the data
    if qb_name not in qb_to_idx:
        raise ValueError(f"Quarterback {qb_name} not found in data.")
    if def_team not in team_to_idx:
        raise ValueError(f"Defense team {def_team} not found in data.")
    
    # Find the latest game index for the QB
    qb_games = df[df['passer_player_name'] == qb_name]
    if qb_games.empty:
        raise ValueError(f"No games found for QB: {qb_name}")
    latest_game_idx = qb_games.index[-1]
    
    # Create sequences
    qb_seq = create_sequence_features(df, qb_name, latest_game_idx)
    def_seq = create_defense_sequence(df, def_team, latest_game_idx)
    pbp_seq = create_play_by_play_sequences(play_by_play, qb_name, qb_games.iloc[-1]['game_id'])
    
    # Convert to tensors
    qb_seq_tensor = torch.FloatTensor(qb_seq).unsqueeze(0)  # Add batch dimension
    def_seq_tensor = torch.FloatTensor(def_seq).unsqueeze(0)
    pbp_seq_tensor = torch.FloatTensor(pbp_seq).unsqueeze(0)
    
    # Get indices
    qb_idx = torch.LongTensor([qb_to_idx[qb_name]])
    team_idx = torch.LongTensor([team_to_idx[def_team]])
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(qb_seq_tensor, def_seq_tensor, pbp_seq_tensor, qb_idx, team_idx)
    
    # Inverse transform the prediction
    prediction = scaler.inverse_transform(prediction.numpy())
    
    # Return the prediction as a dictionary
    return {
        'yards_gained': prediction[0, 0],
        'pass_touchdown': prediction[0, 1],
        'interception': prediction[0, 2],
        'completion_percentage': prediction[0, 3],
        'sack': prediction[0, 4]
    }

# Example usage
try:
    prediction = predict_qb_performance("P.Mahomes", "BUF")
    print("Predicted QB Performance:")
    print(prediction)
except ValueError as e:
    print(e)


