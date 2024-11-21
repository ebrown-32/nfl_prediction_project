import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import nfl_data_py as nfl
from tabulate import tabulate
import datetime

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
    
    if len(sequence) > 0:
        weights = np.exp(np.linspace(-2, 0, len(sequence)))
        sequence = sequence * weights[:, np.newaxis]
    
    if len(sequence) < max_history:
        padding = np.zeros((max_history - len(sequence), 16))
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
        row['pass_attempt'],
        row['air_yards'],
        row['yards_after_catch'],
        row['qb_hit'],
        row['sack'],
        row['short_passes'],
        row['deep_passes'],
        row['left_passes'],
        row['middle_passes'],
        row['right_passes'],
        row['completion_percentage'],
        row['yards_per_attempt'],
        row['sack_rate'],
        row['deep_pass_rate']
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

class QBPerformancePredictor(nn.Module):
    def __init__(self, num_qbs, num_teams, qb_seq_length=8, def_seq_length=4):
        super().__init__()
        
        # Feature dimensions
        self.qb_feature_dim = 16   
        self.def_feature_dim = 11  
        self.hidden_dim = 64
        
        # Sequence processing
        self.qb_norm = nn.LayerNorm([qb_seq_length, self.qb_feature_dim])
        self.def_norm = nn.LayerNorm([def_seq_length, self.def_feature_dim])
        
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
        
        # Embeddings
        self.qb_embedding = nn.Embedding(num_qbs, 16)
        self.team_embedding = nn.Embedding(num_teams, 16)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2*self.hidden_dim + 32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 17)  # Updated to output 17 values
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, qb_seq, def_seq, qb_idx, team_idx):
        # Normalize sequences
        qb_seq = self.qb_norm(qb_seq)
        def_seq = self.def_norm(def_seq)
        
        # Process sequences
        qb_out, _ = self.qb_lstm(qb_seq)
        def_out, _ = self.def_lstm(def_seq)
        
        # Get final states
        qb_final = qb_out[:, -1, :]
        def_final = def_out[:, -1, :]
        
        # Get embeddings
        qb_emb = self.qb_embedding(qb_idx)
        team_emb = self.team_embedding(team_idx)
        
        # Concatenate all features
        combined = torch.cat([qb_final, def_final, qb_emb, team_emb], dim=1)
        
        # Final prediction
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

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
        'yards': predictions[0][0],
        'touchdowns': predictions[0][1],
        'interception': predictions[0][2],
        'attempts': predictions[0][3],
        'air_yards': predictions[0][4],
        'yards_after_catch': predictions[0][5],
        'qb_hits': predictions[0][6],
        'sacks': predictions[0][7],
        'short_passes': predictions[0][8],
        'deep_passes': predictions[0][9],
        'left_passes': predictions[0][10],
        'middle_passes': predictions[0][11],
        'right_passes': predictions[0][12],
        'completion_pct': predictions[0][13],
        'yards_per_attempt': predictions[0][14],
        'sack_rate': predictions[0][15],
        'deep_pass_rate': predictions[0][16]
    }

def save_predictions_to_file(predictions_list, filename=None):
    """
    Save predictions to a formatted text file with raw values
    """
    if filename is None:
        date = datetime.datetime.now().strftime("%Y%m%d")
        filename = f"qb_predictions_{date}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"QB Performance Predictions - Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for pred in predictions_list:
            qb_name = pred['qb']
            opponent = pred['opponent']
            stats = pred['stats']
            
            f.write(f"\n{qb_name} vs {opponent}\n")
            f.write("=" * 50 + "\n\n")
            
            # Core Stats Table
            core_stats = [
                ["Yards", stats['yards']],
                ["Touchdowns", stats['touchdowns']],
                ["Interception", stats['interception']],
                ["Attempts", stats['attempts']],
                ["Completion %", stats['completion_pct']],
                ["Yards/Attempt", stats['yards_per_attempt']]
            ]
            f.write("Core Stats:\n")
            f.write(tabulate(core_stats, tablefmt="grid") + "\n\n")
            
            # Pass Distribution Table
            pass_dist = [
                ["Short Passes", stats['short_passes']],
                ["Deep Passes", stats['deep_passes']],
                ["Deep Pass Rate", stats['deep_pass_rate']],
                ["Left", stats['left_passes']],
                ["Middle", stats['middle_passes']],
                ["Right", stats['right_passes']]
            ]
            f.write("Pass Distribution:\n")
            f.write(tabulate(pass_dist, tablefmt="grid") + "\n\n")
            
            # Protection Table
            protection = [
                ["QB Hits", stats['qb_hits']],
                ["Sacks", stats['sacks']],
                ["Sack Rate", stats['sack_rate']]
            ]
            f.write("Protection Metrics:\n")
            f.write(tabulate(protection, tablefmt="grid") + "\n\n")
            
            # Yardage Table
            yardage = [
                ["Air Yards (Avg.)", stats['air_yards']],
                ["Yards After Catch (Avg.)", stats['yards_after_catch']]
            ]
            f.write("Yardage Breakdown:\n")
            f.write(tabulate(yardage, tablefmt="grid") + "\n\n")
            f.write("\n" + "-" * 50 + "\n")

# Create predictions list
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
    }
]

# Save predictions to file
save_predictions_to_file(predictions_list)

# Still print to console as well
print("\nPredictions have been saved to file!")
print("Example prediction:")
prediction = predictions_list[0]['stats']
print("\nCore Stats:")
print(f"Yards: {prediction['yards']}")
print(f"Touchdowns: {prediction['touchdowns']}")
print(f"Interceptions: {prediction['interception']}")


