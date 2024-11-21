import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nfl_data_py as nfl

years = [2022, 2023, 2024]

play_by_play = pd.DataFrame
play_by_play = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)
play_by_play.to_csv("pbp_data.csv") 

df = pd.read_csv('pbp_data.csv')

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

game_data.to_csv('game_data.csv', index=False)


# Load data
df = pd.read_csv('game_data.csv')

# Calculate historical averages for QBs
qb_historical = df.groupby('passer_player_name').agg({
    'yards_gained': 'mean',
    'pass_touchdown': 'mean',
    'pass_attempt': 'mean',
    'air_yards': 'mean',
    'yards_after_catch': 'mean',
    'shotgun': 'mean'
}).add_prefix('hist_')

# Calculate historical averages for defenses
def_historical = df.groupby('defteam').agg({
    'yards_gained': 'mean',
    'pass_touchdown': 'mean',
    'pass_attempt': 'mean',
    'air_yards': 'mean',
    'yards_after_catch': 'mean'
}).add_prefix('def_hist_')

# Calculate rolling averages for QBs (last 5 games)
df = df.sort_values(['passer_player_name', 'game_id'])
rolling_stats = ['yards_gained', 'pass_touchdown', 'pass_attempt', 'air_yards', 'yards_after_catch']
for stat in rolling_stats:
    df[f'rolling_{stat}'] = df.groupby('passer_player_name')[stat].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

# Calculate defense rolling averages (last 5 games)
df = df.sort_values(['defteam', 'game_id'])
for stat in rolling_stats:
    df[f'def_rolling_{stat}'] = df.groupby('defteam')[stat].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

# Add historical averages to dataframe
df = df.merge(qb_historical, left_on='passer_player_name', right_index=True)
df = df.merge(def_historical, left_on='defteam', right_index=True)

# Create features
feature_columns = [
    # Historical QB performance
    'hist_yards_gained',
    'hist_pass_touchdown',
    'hist_pass_attempt',
    'hist_air_yards',
    'hist_yards_after_catch',
    'hist_shotgun',
    
    # Historical defense performance
    'def_hist_yards_gained',
    'def_hist_pass_touchdown',
    'def_hist_pass_attempt',
    'def_hist_air_yards',
    'def_hist_yards_after_catch',
    
    # Recent QB performance
    'rolling_yards_gained',
    'rolling_pass_touchdown',
    'rolling_pass_attempt',
    'rolling_air_yards',
    'rolling_yards_after_catch',
    
    # Recent defense performance
    'def_rolling_yards_gained',
    'def_rolling_pass_touchdown',
    'def_rolling_pass_attempt',
    'def_rolling_air_yards',
    'def_rolling_yards_after_catch',
    
    # Current game stats
    'shotgun'
]

X = df[feature_columns].fillna(0)
y = df[['yards_gained', 'pass_touchdown', 'pass_attempt']]

# Scale features and targets
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

class QBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class QBPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # Historical data branch
        self.historical_branch = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Recent performance branch
        self.recent_branch = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Combine branches
        self.combined = nn.Sequential(
            nn.Linear(128 + 1, 128),  # +1 for current shotgun
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # Split input into historical, recent, and current
        historical = x[:, :11]  # First 11 features are historical
        recent = x[:, 11:-1]    # Next 10 features are recent
        current = x[:, -1].unsqueeze(1)  # Last feature is current shotgun
        
        # Process branches
        h1 = self.historical_branch(historical)
        h2 = self.recent_branch(recent)
        
        # Combine all features
        combined = torch.cat([h1, h2, current], dim=1)
        return self.combined(combined)

# Create data loaders
train_dataset = QBDataset(X_train, y_train)
test_dataset = QBDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, loss function, and optimizer
model = QBPredictor(input_size=len(feature_columns))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            val_loss += criterion(model(X_batch), y_batch).item()
    
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(test_loader):.4f}\n')

def predict_qb_performance(qb_name, opponent_team):
    """
    Predict QB performance given QB name and opponent team.
    """
    # Get historical averages
    qb_hist = qb_historical.loc[qb_name]
    def_hist = def_historical.loc[opponent_team]
    
    # Get recent performance
    qb_recent = df[df['passer_player_name'] == qb_name].iloc[-1]
    def_recent = df[df['defteam'] == opponent_team].iloc[-1]
    
    # Create feature vector
    features = pd.DataFrame({
        # Historical stats
        'hist_yards_gained': [qb_hist['hist_yards_gained']],
        'hist_pass_touchdown': [qb_hist['hist_pass_touchdown']],
        'hist_pass_attempt': [qb_hist['hist_pass_attempt']],
        'hist_air_yards': [qb_hist['hist_air_yards']],
        'hist_yards_after_catch': [qb_hist['hist_yards_after_catch']],
        'hist_shotgun': [qb_hist['hist_shotgun']],
        'def_hist_yards_gained': [def_hist['def_hist_yards_gained']],
        'def_hist_pass_touchdown': [def_hist['def_hist_pass_touchdown']],
        'def_hist_pass_attempt': [def_hist['def_hist_pass_attempt']],
        'def_hist_air_yards': [def_hist['def_hist_air_yards']],
        'def_hist_yards_after_catch': [def_hist['def_hist_yards_after_catch']],
        
        # Recent performance
        'rolling_yards_gained': [qb_recent['rolling_yards_gained']],
        'rolling_pass_touchdown': [qb_recent['rolling_pass_touchdown']],
        'rolling_pass_attempt': [qb_recent['rolling_pass_attempt']],
        'rolling_air_yards': [qb_recent['rolling_air_yards']],
        'rolling_yards_after_catch': [qb_recent['rolling_yards_after_catch']],
        'def_rolling_yards_gained': [def_recent['def_rolling_yards_gained']],
        'def_rolling_pass_touchdown': [def_recent['def_rolling_pass_touchdown']],
        'def_rolling_pass_attempt': [def_recent['def_rolling_pass_attempt']],
        'def_rolling_air_yards': [def_recent['def_rolling_air_yards']],
        'def_rolling_yards_after_catch': [def_recent['def_rolling_yards_after_catch']],
        
        # Current game
        'shotgun': [qb_recent['shotgun']]
    })
    
    # Scale features
    features_scaled = X_scaler.transform(features)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        scaled_pred = model(torch.FloatTensor(features_scaled))
        predictions = y_scaler.inverse_transform(scaled_pred)
    
    return {
        'yards': round(predictions[0][0], 1),
        'touchdowns': round(predictions[0][1], 2),
        'attempts': round(predictions[0][2], 1)
    }

# Example usage
print("\nExample Predictions:")
print("Trevor Lawrence vs Chiefs:")
print(predict_qb_performance("T.Lawrence", "KC"))
print("\nPatrick Mahomes vs Bills:")
print(predict_qb_performance("P.Mahomes", "BUF"))
print(predict_qb_performance("P.Mahomes", "CIN"))
print(predict_qb_performance("P.Mahomes", "LV"))
print(predict_qb_performance("P.Mahomes", "DEN"))
print(predict_qb_performance("J.Burrow", "LAC"))