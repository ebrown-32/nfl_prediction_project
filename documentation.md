# QB Performance Predictor: Technical Documentation

## Data Architecture

### Input Features (18 dimensions per play)
1. **Core Play Statistics**
   - Yards gained
   - Pass touchdown (binary)
   - Complete pass (binary)
   - Air yards
   - Yards after catch
   - QB hit (binary)
   - Sack (binary)

2. **Game Situation**
   - Score differential
   - Quarter
   - Down
   - Yards to go
   - Field position (yards from opponent's end zone)

3. **Defensive Context**
   - Defenders in box
   - Number of pass rushers

4. **Environmental Factors**
   - Temperature
   - Wind speed

5. **Formation**
   - Shotgun (binary)
   - No huddle (binary)

### Sequence Processing
- Each QB and defense has a sequence of plays represented as an `N x 18` matrix, where N is variable
- Recent plays are weighted more heavily using linear weighting: `weights = np.linspace(1.0, 1.5, len(sequence))`
- Maximum sequence length is capped at 2000 plays
- Sequences are padded with zeros if shorter than the maximum length

## Model Architecture Details

### 1. Feature Embedding Layer
```python
self.feature_embedding = nn.Linear(18, 64)
```

- Transforms raw 18-dimensional features into 64-dimensional embeddings
- Allows model to learn higher-level feature representations

### 2. Transformer Encoder
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=64,      # Embedding dimension
    nhead=4,         # Number of attention heads
    dim_feedforward=256,
    dropout=0.1,
    batch_first=True
)
self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
```

- Uses 3 transformer encoder layers
- Each layer has 4 attention heads for multi-head attention
- Feedforward network dimension of 256
- 10% dropout for regularization

### 3. Identity Embeddings
```python
self.qb_embedding = nn.Embedding(num_qbs, 32)
self.team_embedding = nn.Embedding(num_teams, 32)
```

- Each QB and team gets a unique 32-dimensional embedding
- Learned during training to capture player/team characteristics

### 4. Positional Encoding
```python
self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, 64))
```

- Learnable positional encodings
- Added to embedded sequences to maintain temporal information
- Shape: `[1, 2000, 64]`

### 5. Attention Pooling
```python
def attention_pool(self, x):
    weights = torch.softmax(self.attention_weights(x), dim=1)
    return torch.sum(weights * x, dim=1)
```

- Computes attention weights for each timestep
- Weighted sum reduces sequence to single vector
- Allows model to focus on most relevant plays

### 6. Output Networks
```python
# Main output
self.fc1 = nn.Linear(192, 128)  # 192 = 64*2 + 32*2 (pooled sequences + embeddings)
self.fc2 = nn.Linear(128, 64)
self.fc3 = nn.Linear(64, 5)     # 5 output metrics

# Auxiliary output
self.aux_fc1 = nn.Linear(192, 64)
self.aux_fc2 = nn.Linear(64, 3)  # 3 auxiliary metrics
```

## Training Details

### Loss Functions
1. **Main Loss**: Focal Loss
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.mse_loss(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

- Reduces impact of easy examples
- Focuses training on harder cases
- Parameters: α=1, γ=2

2. **Auxiliary Loss**: MSE Loss
- Used for completion percentage, TD rate, and INT rate
- Combined with main loss using 0.3 weight factor

### Optimization
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2
)
```

- AdamW optimizer with weight decay
- Cosine annealing learning rate schedule
- Initial learning rate: 0.001
- Weight decay: 0.01

### Training Process
- Batch size: 16
- Early stopping with patience of 7 epochs
- Gradient clipping at 0.5
- Curriculum learning with increasing sequence lengths
- Layer normalization and dropout (0.2) for regularization

## Output Metrics
1. **Main Predictions**
   - Yards gained (continuous)
   - Pass touchdowns (continuous)
   - Interceptions (continuous)
   - Completion percentage (continuous)
   - Sacks (continuous)

2. **Auxiliary Predictions**
   - Completion percentage
   - Touchdown rate
   - Interception rate

# QB Performance Predictor: Technical Documentation

## Data Flow and Play Mapping

### Play-to-Sequence Mapping
1. **Individual Play Processing**
   - Each play is processed into an 18-dimensional feature vector
   - Features are normalized using StandardScaler
   - Missing values are filled with sensible defaults (0 for binary features, medians for continuous)

2. **QB Sequence Creation**
```python
def create_sequence_features(play_by_play, qb_name, game_id):
    """Creates a sequence of play-by-play data for a specific QB"""
    # Filter plays for the QB up to the specified game
    qb_plays = play_by_play[
        (play_by_play['passer_player_name'] == qb_name) & 
        (play_by_play['game_id'] < game_id)
    ].sort_values('game_id')
    
    sequence = []
    for _, play in qb_plays.iterrows():
        play_stats = [
            play['yards_gained'],
            play['pass_touchdown'],
            play['complete_pass'],
            play['air_yards'],
            play['yards_after_catch'],
            play['qb_hit'],
            play['sack'],
            play['score_differential'],
            play['quarter'],
            play['down'],
            play['yards_to_go'],
            play['yardline_100'],
            play['defenders_in_box'],
            play['number_of_pass_rushers'],
            play['temperature'],
            play['wind_speed'],
            play['shotgun'],
            play['no_huddle']
        ]
        sequence.append(play_stats)
    return np.array(sequence)
```

3. **Defense Sequence Creation**
```python
def create_defense_sequence(play_by_play, defense_team, game_id):
    """Creates a sequence of play-by-play data for a specific defense"""
    # Filter plays against the defense up to the specified game
    def_plays = play_by_play[
        (play_by_play['defteam'] == defense_team) & 
        (play_by_play['game_id'] < game_id)
    ].sort_values('game_id')
    
    # Similar feature extraction as QB sequence
    sequence = []
    for _, play in def_plays.iterrows():
        play_stats = [...]  # Same 18 features
        sequence.append(play_stats)
    return np.array(sequence)
```

### Sequence Management
```
Input Play Data
      │
      ▼
┌─────────────┐
│ Play Filter │
└─────────────┘
      │
      ▼
┌─────────────────────┐
│ Feature Extraction  │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Sequence Formation  │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Recent Play Weight  │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Length Management   │
└─────────────────────┘
      │
      ▼
Final Sequence
```

## Model Architecture Diagram
```
Input Features (18-dim)
         │
         ▼
┌─────────────────────┐
│  Feature Embedding  │
│      (18 → 64)     │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│     Positional      │
│     Encoding        │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    Transformer      │
│     Encoder         │──┐
│   (3 layers, 4      │  │
│  attention heads)   │  │
└─────────────────────┘  │
         │              │
         ▼              │
┌─────────────────────┐ │
│ Attention Pooling   │ │
└─────────────────────┘ │
         │              │
         ▼              │
┌─────────────────────┐ │
│ Identity Embeddings │ │
│    (QB + Team)      │ │
└─────────────────────┘ │
         │              │
         ▼              ▼
┌─────────────────────────────┐
│      Output Networks        │
├─────────────────────────────┤
│ Main (5 metrics) │ Aux (3) │
└─────────────────────────────┘
```

### Sequence Length Management
1. **Maximum Length**: 2000 plays
   - Longer sequences are truncated from the start
   - More recent plays are preserved
   - Ensures computational efficiency

2. **Minimum Length**: No minimum
   - Short sequences are zero-padded
   - Attention mechanism handles variable lengths
   - Positional encoding maintains temporal information

3. **Weighting**
   - Recent plays weighted more heavily
   - Linear weighting from 1.0 to 1.5
   - Weights applied before sequence truncation

# Play-to-Game Mapping Documentation

## Non-Technical Explanation

Imagine you're trying to predict how well Patrick Mahomes will play in an upcoming game against the Bills. You would:
1. Look at all of Mahomes' previous plays this season
2. Look at all plays against the Bills' defense this season
3. Use this history to predict Mahomes' stats in the upcoming game

For each historical game in our dataset:
- INPUT: All plays by that QB and against that defense from before the game
- OUTPUT: The QB's actual stats from that game

For example:
```
Game: Chiefs vs Bills (Week 6, 2023)
Input:
  - All Mahomes plays from Weeks 1-5
  - All plays against Bills defense from Weeks 1-5
Target:
  - Mahomes' actual stats from the Week 6 game
```

## Technical Explanation

### 1. Data Organization
```python
# First, group all plays by game to get game-level stats
game_stats = pass_plays.groupby(['game_id', 'passer_player_name', 'defteam']).agg({
    'yards_gained': 'sum',
    'pass_touchdown': 'sum',
    'interception': 'sum',
    'complete_pass': 'sum',
    'pass_attempt': 'sum',
    'sack': 'sum'
})

# For each game:
for game_id, game in game_stats.iterrows():
    # Get all plays BEFORE this game
    qb_sequence = plays[
        (plays['passer_player_name'] == game['passer_player_name']) &
        (plays['game_id'] < game['game_id'])
    ]
    
    def_sequence = plays[
        (plays['defteam'] == game['defteam']) &
        (plays['game_id'] < game['game_id'])
    ]
    
    # Target is the actual stats FROM this game
    target = [
        game['yards_gained'],
        game['pass_touchdown'],
        game['interception'],
        game['completion_percentage'],
        game['sack']
    ]
```

### 2. Training Example Structure

Each training example consists of:

**Inputs:**
- QB sequence: `[num_plays x 18]` matrix of historical plays
- Defense sequence: `[num_plays x 18]` matrix of historical plays
- QB index: Integer identifying the quarterback
- Team index: Integer identifying the defense

**Target:**
- 5-dimensional vector of game stats:
  1. Yards gained
  2. Pass touchdowns
  3. Interceptions
  4. Completion percentage
  5. Sacks

### 3. Visual Example

```
Game ID: 2023_KC_BUF_W6
┌─────────────────────────────┐
│ INPUT                       │
├─────────────────────────────┤
│ QB Sequence (Mahomes)       │
│ ├── Week 1 plays           │
│ ├── Week 2 plays           │
│ ├── Week 3 plays           │
│ ├── Week 4 plays           │
│ └── Week 5 plays           │
│                             │
│ Defense Sequence (Bills)    │
│ ├── Week 1 plays           │
│ ├── Week 2 plays           │
│ ├── Week 3 plays           │
│ ├── Week 4 plays           │
│ └── Week 5 plays           │
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ TARGET                      │
├─────────────────────────────┤
│ Week 6 Game Stats           │
│ ├── 315 yards              │
│ ├── 3 touchdowns           │
│ ├── 0 interceptions        │
│ ├── 68.4% completion       │
│ └── 1 sack                 │
└─────────────────────────────┘
```

### 4. Key Points

1. **Temporal Ordering**
   - Only plays that occurred before a game are used to predict that game's stats
   - This prevents data leakage and mimics real-world prediction scenarios

2. **Sequence Processing**
   - Recent plays are weighted more heavily (1.0 to 1.5 linear scaling)
   - Sequences are capped at 2000 plays
   - Shorter sequences are padded with zeros

3. **Game-Level Aggregation**
   - Individual plays are never predicted
   - Model learns to map sequences directly to game-level statistics
   - This approach captures both recent form and matchup dynamics