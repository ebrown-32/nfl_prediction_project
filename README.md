# NFL Quarterback Performance Predictor

## Overview
This deep learning model predicts NFL quarterback performance statistics for hypothetical matchups using historical game data and advanced sequence processing techniques. The model combines historical performance data, quarterback-specific characteristics, and defensive matchup information to generate comprehensive statistical predictions.

## Table of Contents
1. [Mathematical Framework](#mathematical-framework)
2. [Model Architecture](#model-architecture)
3. [Data Processing](#data-processing)
4. [Usage](#usage)
5. [Model Evaluation](#model-evaluation)
6. [Installation](#installation)
7. [Future Improvements](#future-improvements)

## Mathematical Framework

### Input Components

The model processes four key input components:

#### 1. QB Historical Sequence (X_q)
- Dimension: ℝ^(16×16) (16 games × 16 features)
- Features include: yards, touchdowns, attempts, completion percentage, etc.
- Exponentially weighted for recency:
  ```
  w_t = exp(-λ(T-t))
  where:
  - t is the game index
  - T is the most recent game
  - λ is the decay factor
  ```

#### 2. Defense Historical Sequence (X_d)
- Dimension: ℝ^(8×11) (8 games × 11 features)
- Captures defensive performance patterns
- Similarly weighted for recency

#### 3. QB Identity Embedding (E_q)
- Dimension: ℝ⁶⁴
- Learned representation of quarterback-specific traits
- Maps discrete QB identity to continuous vector space

#### 4. Team Identity Embedding (E_d)
- Dimension: ℝ³²
- Learned representation of defensive team characteristics

### LSTM Architecture

The model uses Long Short-Term Memory (LSTM) networks to process sequential data. The LSTM cell is defined by:

```
f_t = σ(W_f · [h_t-1, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_t-1, x_t] + b_i)    # Input gate
C̃_t = tanh(W_c · [h_t-1, x_t] + b_c) # Candidate cell state
C_t = f_t * C_t-1 + i_t * C̃_t        # Cell state
o_t = σ(W_o · [h_t-1, x_t] + b_o)    # Output gate
h_t = o_t * tanh(C_t)                # Hidden state
```

Where:
- h_t is the hidden state at time t
- C_t is the cell state at time t
- x_t is the input at time t
- W and b are learned parameters
- σ is the sigmoid function

### Prediction Process

1. **Sequence Normalization**:
   ```
   X̂_q = LayerNorm(X_q)
   X̂_d = LayerNorm(X_d)
   ```

2. **LSTM Processing**:
   ```
   h_q = LSTM(X̂_q)
   h_d = LSTM(X̂_d)
   ```

3. **Feature Combination**:
   ```
   z = concat[h_q, h_d, E_q, E_d]
   ```

4. **Final Prediction**:
   ```
   y = MLP(z)
   ```
   where y ∈ ℝ¹⁷ represents 17 predicted statistical categories

## Model Architecture
Input → Normalization → LSTM Processing → Embeddings → Attention → Dense Layers → Prediction

QB Data    →  Norm → LSTM(2) ↘
                            → Concat → Attention → Dense(3) → Output
Defense Data → Norm → LSTM(2) ↗

Total Layer Count: 12 main layers
Input Normalization (2)
QB LSTM Stack (2)
Defense LSTM Stack (2)
Embeddings (2)
Attention Layer (1)
QB-specific Layer (1)
Final Dense Layers (3)

### Network Components

1. **Layer Normalization**
   - Stabilizes learning by normalizing input sequences
   - Applied to both QB and defensive sequences

2. **LSTM Layers**
   - 2 stacked LSTM layers for both QB and defense sequences
   - Hidden dimension: 64
   - Dropout: 0.1
   - Batch-first processing

3. **Attention Mechanism**
   - Multi-head attention (4 heads)
   - Helps focus on relevant historical patterns
   - Dimension: 64

4. **Fully Connected Layers**
   - FC1: Input → 64 units
   - FC2: 64 → 32 units
   - FC3: 32 → 17 units (final predictions)
   - ReLU activation between layers
   - Dropout: 0.1

## Data Processing

### Input Data
- Source: NFL play-by-play data (2022-2024) from nfl-data-py (https://pypi.org/project/nfl-data-py/)
- Aggregated to game level
- Features normalized using StandardScaler
- Sequences padded to fixed length

### Sequence Creation
```python
def create_sequence_features(df, qb_name, game_idx, max_history=16):
    # Get QB's previous games
    qb_games = df[df['passer_player_name'] == qb_name]
    previous_games = qb_games[qb_games.index < game_idx].tail(max_history)
    
    # Create and weight sequence
    sequence = []
    for _, game in previous_games.iterrows():
        game_stats = [
            game['yards_gained'],
            game['pass_touchdown'],
            # ... other stats
        ]
        sequence.append(game_stats)
```

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Making Predictions
```python
# Initialize model
model = QBPerformancePredictor(
    num_qbs=len(qb_to_idx),
    num_teams=len(team_to_idx)
)

# Make prediction
predictions = predict_qb_performance(
    qb_name="Patrick Mahomes",
    opponent_team="Bills"
)

# Save predictions
save_predictions_to_file(predictions_list)
```

### Output Format
```python
{
    'yards': 285.5,
    'touchdowns': 2.3,
    'interception': 0.8,
    'attempts': 35.2,
    # ... other stats
}
```

## Model Evaluation

### Training Process
- Batch size: 16
- Learning rate: 0.0001
- Weight decay: 1e-5
- Early stopping patience: 10
- Train/Test split: 80/20

### Metrics
- Mean Squared Error (MSE) for continuous statistics
- Validation loss monitoring
- Gradient clipping at 0.5

### Error Handling
- NaN detection and cleaning
- Gradient explosion prevention
- Sequence padding management

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- scikit-learn

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the data_preparation.py script to generate the 'game_data.csv' file that the model will use to train.
4. Train the model and make predictions with the nn.py script!

## Future Improvements

1. **Data Enhancements**
   - Add weather condition effects
   - Incorporate offensive line performance
   - Add receiver corps quality metrics
   - Include home/away factors

2. **Model Architecture**
   - Implement transformer architecture
   - Add positional encoding
   - Explore different attention mechanisms
   - Implement hierarchical LSTM

3. **Feature Engineering**
   - Create advanced defensive metrics
   - Add game context features
   - Develop QB style classifications

4. **Prediction Capabilities**
   - Add confidence intervals
   - Implement ensemble methods
   - Add probability distributions for predictions