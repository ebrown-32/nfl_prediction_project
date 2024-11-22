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

#TODO:
# - Add weather conditions
# - Add home/away factor
# - Add formations
# - Add receiver corps quality metrics
# Adjust recent game weighting... maybe linear is better?

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

def create_sequence_features(df, qb_name, game_idx, max_history=16):
    """
    Creates a sequence of game statistics for a quarterback's recent performances.
    
    Sequence Processing Explanation:
    - Design decision: the order of games matters (recent form, development, etc.)
    - We take the last 16 games (max_history) for each QB
    - More recent games are weighted more heavily using exponential weighting
    
    Example:
    For Patrick Mahomes predicting Week 10:
    - Week 9: weight = 1.0
    - Week 8: weight = 0.87
    - Week 7: weight = 0.76
    ...and so on, giving more importance to recent performances
    """
    qb_games = df[df['passer_player_name'] == qb_name]
    previous_games = qb_games[qb_games.index < game_idx].tail(max_history)
    
    # Create sequence with exponential weighting
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
            game['short_passes'] if not np.isnan(game['short_passes']) else 0,
            game['deep_passes'] if not np.isnan(game['deep_passes']) else 0,
            game['left_passes'] if not np.isnan(game['left_passes']) else 0,
            game['middle_passes'] if not np.isnan(game['middle_passes']) else 0,
            game['right_passes'] if not np.isnan(game['right_passes']) else 0,
            game['completion_percentage'] if not np.isnan(game['completion_percentage']) else 0,
            game['yards_per_attempt'] if not np.isnan(game['yards_per_attempt']) else 0,
            game['sack_rate'] if not np.isnan(game['sack_rate']) else 0,
            game['deep_pass_rate'] if not np.isnan(game['deep_pass_rate']) else 0
        ]
        sequence.append(game_stats)
    
    sequence = np.array(sequence)
    
    # Apply exponential weighting to emphasize recent games
    if len(sequence) > 0:
        weights = np.exp(np.linspace(-2, 0, len(sequence)))
        sequence = sequence * weights[:, np.newaxis]
    
    if len(sequence) < max_history:
        padding = np.zeros((max_history - len(sequence), 16))
        sequence = np.vstack([padding, sequence]) if len(sequence) > 0 else padding
    
    return sequence

def create_defense_sequence(df, def_team, game_idx, max_history=8):
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
            game['yards_after_catch'] if not np.isnan(game['yards_after_catch']) else 0,
            game['qb_hit'] if not np.isnan(game['qb_hit']) else 0,
            game['sack'] if not np.isnan(game['sack']) else 0,
            game['short_passes'] if not np.isnan(game['short_passes']) else 0,
            game['deep_passes'] if not np.isnan(game['deep_passes']) else 0,
            game['completion_percentage'] if not np.isnan(game['completion_percentage']) else 0,
            game['sack_rate'] if not np.isnan(game['sack_rate']) else 0
        ]
        sequence.append(game_stats)
    
    sequence = np.array(sequence)
    
    if len(sequence) > 0:
        weights = np.exp(np.linspace(-1.5, 0, len(sequence)))
        sequence = sequence * weights[:, np.newaxis]
    
    if len(sequence) < max_history:
        padding = np.zeros((max_history - len(sequence), 11))
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
        row['interception'],
        row['completion_percentage'],
        row['yards_per_attempt']
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

""" 
Each index has a unique integer value mapped to a string (name of QB or team)

# Example of what qb_to_idx might look like:
qb_to_idx = {
    "P.Mahomes": 0,
    "J.Allen": 1,
    "L.Jackson": 2,
    # ... and so on
}

# Example of what team_to_idx might look like:
team_to_idx = {
    "KC": 0,
    "BUF": 1,
    "BAL": 2,
    # ... and so on
} """

class NFLDataset(Dataset):
    def __init__(self, X_qb, X_def, y, qb_names, def_teams, indices):
        """
        Parameters:
        - X_qb: QB sequence data
        - X_def: Defensive team sequence data
        - y: Target statistics
        - qb_names: Names of QBs (e.g., "P.Mahomes")
        - def_teams: Names of defensive teams (e.g., "BUF" when predicting Mahomes vs Bills)
        - indices: Which samples to include in this dataset
        """
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

class QBPerformancePredictor(nn.Module):
    """
    Neural network using LSTM (Long Short-Term Memory) architecture for QB prediction.
    
    https://en.wikipedia.org/wiki/Long_short-term_memory
    https://en.wikipedia.org/wiki/Recurrent_neural_network
    https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/300px-LSTM_Cell.svg.png
    
    LSTM Explanation:
    - LSTM is a type of RNN (Recurrent Neural Network) that can learn long-term dependencies
    - Unlike simple RNNs, LSTMs can "remember" important information and "forget" irrelevant details
    
    LSTM Components:
    1. Forget Gate (f_t): Decides what information to throw away
       f_t = σ(W_f · [h_t-1, x_t] + b_f)
    
    2. Input Gate (i_t): Decides what new information to store
       i_t = σ(W_i · [h_t-1, x_t] + b_i)
    
    3. Cell State (C_t): The memory of the network
       C_t = f_t * C_t-1 + i_t * tanh(W_c · [h_t-1, x_t] + b_c)
    
    4. Output Gate (o_t): Decides what parts of the cell state to output
       o_t = σ(W_o · [h_t-1, x_t] + b_o)
    
    In QB Context:
    - Input sequence: Last 16 games of QB stats
    - Cell state: Maintains important long-term performance patterns
    - Gates: Learn what patterns are predictive of future performance
    """
    def __init__(self, num_qbs, num_teams, qb_seq_length=16, def_seq_length=8):
        super().__init__()
        
        self.qb_feature_dim = 16 
        self.def_feature_dim = 11 
        self.hidden_dim = 64 
        
        # Layer normalization for input sequences
        self.qb_norm = nn.LayerNorm([qb_seq_length, self.qb_feature_dim])
        self.def_norm = nn.LayerNorm([def_seq_length, self.def_feature_dim])
        
        # LSTM layers for sequence processing
        # QB LSTM: Processes historical game sequences with exponentially weighted recency
        self.qb_lstm = nn.LSTM(
            input_size=self.qb_feature_dim,    # Number of stats per game
            hidden_size=self.hidden_dim,       # Size of internal state
            num_layers=2,                      # Stacked LSTMs for more complexity
            batch_first=True,                  # Input shape: (batch, seq, feature)
            dropout=0.1                        # Prevent overfitting
        )
        """
        QB LSTM Process:
        1. Takes sequence of 16 games: X = [x₁, x₂, ..., x₁₆]
        2. Each game xᵢ has features: [yards, touchdowns, attempts, ...]
        3. LSTM processes games in order, updating its state:
           h_t = LSTM(x_t, h_{t-1})
        4. Final state h₁₆ contains summary of QB's recent performance trends
        """
        
        # Defense LSTM: Processes opponent's defensive performance history
        self.def_lstm = nn.LSTM(
            input_size=self.def_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Identity embeddings
        # Maps discrete QB/Team identities to continuous vectors
        self.qb_embedding_dim = 64    # Higher dim for QB (more complex patterns)
        self.team_embedding_dim = 32   # Lower dim for defense (less complex patterns)
        
        self.qb_embedding = nn.Embedding(num_qbs, self.qb_embedding_dim)
        self.team_embedding = nn.Embedding(num_teams, self.team_embedding_dim)
        
        # QB-specific attention for focusing on relevant historical patterns
        self.qb_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4)
        
        # Conditional processing based on QB identity
        self.qb_specific_layer = nn.Linear(self.qb_embedding_dim, self.hidden_dim)
        
        # Fully connected layers for final prediction
        # Input: concatenated [QB sequence, Defense sequence, QB embedding, Team embedding]
        fc1_input_size = (2 * self.hidden_dim) + self.qb_embedding_dim + self.team_embedding_dim
        self.fc1 = nn.Linear(fc1_input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)  # Changed from 17 to 5 outputs
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, qb_seq, def_seq, qb_idx, team_idx):
        """
        Forward pass showing how sequences are processed.
        
        Sequence Processing Steps:
        1. Normalize sequences to stabilize learning:
           qb_seq = LayerNorm(qb_seq)
        
        2. Process through LSTM:
           - Input: [batch_size, 16 games, 16 features]
           - LSTM maintains internal state for each game
           - Output: [batch_size, 16 games, hidden_dim]
        
        3. Take final state as sequence summary:
           qb_final = qb_out[:, -1, :]
        
        Example for one QB:
        - Input: Last 16 games of stats
        - LSTM processes each game in order
        - Final output combines recent form, long-term patterns, and trends
        """
        # Process QB sequence through LSTM
        qb_seq = self.qb_norm(qb_seq)
        qb_out, _ = self.qb_lstm(qb_seq)    # Process all 16 games
        qb_final = qb_out[:, -1, :]         # Take final state as summary
        
        # Normalize sequences
        def_seq = self.def_norm(def_seq)
        
        # Process sequences through LSTMs
        qb_out, _ = self.qb_lstm(qb_seq)    # [batch, seq_len, hidden_dim]
        def_out, _ = self.def_lstm(def_seq)  # [batch, seq_len, hidden_dim]
        
        # Get final sequence states
        qb_final = qb_out[:, -1, :]         # [batch, hidden_dim]
        def_final = def_out[:, -1, :]       # [batch, hidden_dim]
        
        # Get identity embeddings
        qb_emb = self.qb_embedding(qb_idx)   # [batch, qb_embedding_dim]
        team_emb = self.team_embedding(team_idx)  # [batch, team_embedding_dim]
        
        # Combine all features
        combined = torch.cat([qb_final, def_final, qb_emb, team_emb], dim=1)
        
        # Final prediction layers
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Final prediction of 17 statistical categories
        
        return x

# Check for NaN values (could indicate problem with data preparation script)
print("\nChecking data for NaN values...")
print(f"X_qb NaN count: {np.isnan(X_qb).sum()}")
print(f"X_def NaN count: {np.isnan(X_def).sum()}")
print(f"y NaN count: {np.isnan(y).sum()}")

X_qb = np.nan_to_num(X_qb, 0)
X_def = np.nan_to_num(X_def, 0)
y = np.nan_to_num(y, 0)

# Create data loaders with smaller batch size
train_dataset = NFLDataset(X_qb, X_def, y_scaled, df['passer_player_name'], df['defteam'], train_idx)
test_dataset = NFLDataset(X_qb, X_def, y_scaled, df['passer_player_name'], df['defteam'], test_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize model, loss function, and optimizer
print("Initializing model...")
model = QBPerformancePredictor(
    num_qbs=len(qb_to_idx),
    num_teams=len(team_to_idx)
)

# Initialize weights properly
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
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

# At the start of training, initialize lists to store losses
train_losses = []
val_losses = []

# Training loop
print("Starting training...")
num_epochs = 100
best_val_loss = float('inf')
patience = 10  # Number of epochs to wait before early stopping
patience_counter = 0

for epoch in range(num_epochs):
    """
    Training Process (per epoch):
    1. Training Phase:
       - Process mini-batches of data
       - Update model weights to minimize loss
       - Track total loss for monitoring
    
    2. Validation Phase:
       - Evaluate model on unseen data
       - Check for improvement
       - Implement early stopping if needed
    """
    model.train()  # Set model to training mode (enables dropout, etc.)
    total_loss = 0
    batch_count = 0
    skipped_batches = 0
    
    # Training Phase
    for qb_seq, def_seq, qb_idx, team_idx, y_batch in train_loader:
        """
        Mini-batch Processing:
        - qb_seq: Sequence of QB's last 16 games [batch_size, 16, 16]
        - def_seq: Sequence of defense's last 8 games [batch_size, 8, 11]
        - qb_idx: QB identity numbers [batch_size]
        - team_idx: Defensive team identity numbers [batch_size]
        - y_batch: Target statistics to predict [batch_size, 17]
        """
        # Check for invalid values
        if torch.isnan(qb_seq).any() or torch.isnan(def_seq).any():
            skipped_batches += 1
            continue
            
        # Forward pass: QB stats = model(QB history, Defense history, QB identity, Defense identity)
        y_pred = model(qb_seq, def_seq, qb_idx, team_idx)
        loss = criterion(y_pred, y_batch)
        
        # Skip bad batches (NaN loss)
        if torch.isnan(loss):
            skipped_batches += 1
            continue
            
        # Backward pass (3 steps)
        optimizer.zero_grad()        # 1. Clear previous gradients
        loss.backward()              # 2. Compute new gradients
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()             # 3. Update model weights
        
        total_loss += loss.item()
        batch_count += 1
    
    # Validation Phase
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)
    val_loss = 0
    val_batch_count = 0
    val_skipped = 0
    
    with torch.no_grad():  # Disable gradient computation for validation
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
    
    # Calculate average losses for epoch
    if batch_count > 0:
        avg_train_loss = total_loss / batch_count
        train_losses.append(avg_train_loss)
    else:
        avg_train_loss = float('nan')
        train_losses.append(float('nan'))
        print(f"Warning: All training batches were skipped in epoch {epoch+1}")
    
    if val_batch_count > 0:
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
    else:
        avg_val_loss = float('nan')
        val_losses.append(float('nan'))
        print(f"Warning: All validation batches were skipped in epoch {epoch+1}")
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f} (processed {batch_count}/{len(train_loader)} batches)')
        print(f'Validation Loss: {avg_val_loss:.4f} (processed {val_batch_count}/{len(test_loader)} batches)')
        if skipped_batches > 0:
            print(f'Skipped {skipped_batches} training batches due to NaN values')
        if val_skipped > 0:
            print(f'Skipped {val_skipped} validation batches due to NaN values')
        print()
    
    # Early stopping logic
    """
    Early Stopping:
    - Track best validation loss seen so far
    - If validation loss doesn't improve for 'patience' epochs, stop training
    - Helps prevent overfitting by stopping when model stops improving
    
    Example:
    Epoch 20: val_loss = 0.5 (best so far) → patience_counter = 0
    Epoch 21: val_loss = 0.6 → patience_counter = 1
    Epoch 22: val_loss = 0.7 → patience_counter = 2
    ...
    Epoch 29: val_loss = 0.8 → patience_counter = 9
    Epoch 30: val_loss = 0.9 → patience_counter = 10 → Stop training
    """
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
        'passing_yards': predictions[0][0],
        'touchdowns': predictions[0][1],
        'interceptions': predictions[0][2],
        'completion_percentage': predictions[0][3],
        'yards_per_attempt': predictions[0][4]
    }

def save_predictions_to_file(predictions_list, filename=None):
    """
    Save predictions to a formatted text file
    """
    if filename is None:
        date = datetime.now().strftime("%Y%m%d%S")
        filename = f"predictions/qb_predictions_{date}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"QB Performance Predictions - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for pred in predictions_list:
            qb_name = pred['qb']
            opponent = pred['opponent']
            stats = pred['stats']
            
            f.write(f"\n{qb_name} vs {opponent}\n")
            f.write("=" * 50 + "\n\n")
            
            # Core Stats Table
            core_stats = [
                ["Passing Yards", f"{stats['passing_yards']:.1f}"],
                ["Touchdowns", f"{stats['touchdowns']:.1f}"],
                ["Interceptions", f"{stats['interceptions']:.1f}"],
                ["Completion %", f"{stats['completion_percentage']:.1f}%"],
                ["Yards/Attempt", f"{stats['yards_per_attempt']:.1f}"]
            ]
            f.write("Core Stats:\n")
            f.write(tabulate(core_stats, tablefmt="grid") + "\n\n")
            f.write("\n" + "-" * 50 + "\n")

# List of QBs and opponents to predict
predictions_list = [
    {
        'qb': "Patrick Mahomes",
        'opponent': "Bills",
        'stats': predict_qb_performance("P.Mahomes", "BUF")
    },
    {
        'qb': "Patrick Mahomes",
        'opponent': "Titans",
        'stats': predict_qb_performance("P.Mahomes", "TEN")
    },
    {
        'qb': "Patrick Mahomes",
        'opponent': "49ers",
        'stats': predict_qb_performance("P.Mahomes", "SF")
    },
    {
        'qb': "Patrick Mahomes",
        'opponent': "Panthers",
        'stats': predict_qb_performance("P.Mahomes", "CAR")
    },
    {
        'qb': "Patrick Mahomes",
        'opponent': "Raiders",
        'stats': predict_qb_performance("P.Mahomes", "LV")
    },
    {
        'qb': "Josh Allen",
        'opponent': "Chiefs",
        'stats': predict_qb_performance("J.Allen", "KC")
    },
    {
        'qb': "Lamar Jackson",
        'opponent': "49ers",
        'stats': predict_qb_performance("L.Jackson", "SF")
    },
    {
        'qb': "Brock Purdy",
        'opponent': "Ravens",
        'stats': predict_qb_performance("B.Purdy", "BAL")
    },
    {
        'qb': "Brock Purdy",
        'opponent': "Panthers",
        'stats': predict_qb_performance("B.Purdy", "CAR")
    },
    {
        'qb': "Russell Wilson",
        'opponent': "Browns",
        'stats': predict_qb_performance("R.Wilson", "CLE")
    },
    {
        'qb': "Jameis Winston",
        'opponent': "Steelers",
        'stats': predict_qb_performance("J.Winston", "PIT")
    }
]

save_predictions_to_file(predictions_list)
print("\nPredictions have been saved to file!")

# After training loop ends, create and save the plot
def plot_training_history(train_losses, val_losses):
    """
    Creates and saves a plot of training history.
    
    Args:
        train_losses (list): Training loss values per epoch
        val_losses (list): Validation loss values per epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    
    plt.title('QB Core Stats Prediction - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add min/max annotations
    min_train = min(train_losses)
    min_val = min(val_losses)
    plt.annotate(f'Min Train: {min_train:.4f}', 
                xy=(train_losses.index(min_train), min_train),
                xytext=(10, 10), textcoords='offset points')
    plt.annotate(f'Min Val: {min_val:.4f}', 
                xy=(val_losses.index(min_val), min_val),
                xytext=(10, -10), textcoords='offset points')
    
    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join('training', f'training_history_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

plot_training_history(train_losses, val_losses)

def save_model(model, filename=None):
    """
    Save the trained model in the training directory
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'qb_predictor_{timestamp}.pth'
    
    model_path = os.path.join('training', filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
save_model(model)


