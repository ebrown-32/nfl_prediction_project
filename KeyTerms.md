# Key Terms and Concepts

## Neural Network Architecture

### LSTM (Long Short-Term Memory)
**Technical**: A type of RNN designed to handle sequential data with gates that regulate information flow
**Simple Explanation**: Think of it like a smart notepad that can:
- Remember important patterns from past games
- Forget irrelevant information
- Decide what's worth paying attention to
Just like how a human scout would remember key trends but forget unimportant details.

### Bidirectional LSTM
**Technical**: LSTM that processes sequences in both forward and backward directions
**Simple Explanation**: Like having two scouts watch the same games:
- One watches games in order (start to end of season)
- One watches games in reverse (end to start of season)
- Both perspectives combined give better insights

### Embeddings
**Technical**: Dense vector representations of discrete entities
**Simple Explanation**: Converting names into numbers in a smart way. Instead of just saying "Patrick Mahomes = 1", we create a detailed numerical "profile" that captures different aspects of his playing style. Like having multiple scores for different skills.

Example:
QB Name → Integer → Learned Embedding Vector
"P.Mahomes" → 0 → [0.75, 0.82, -0.23, ..., 0.45]  # 64 numbers
"J.Allen"   → 1 → [0.62, -0.45, 0.31, ..., 0.28]  # 64 numbers
"L.Jackson" → 2 → [0.91, 0.15, 0.44, ..., -0.12]  # 64 numbers

### Layer Normalization
**Technical**: Normalizes inputs across features to stabilize training
**Simple Explanation**: Like grading on a curve in school - converts all numbers to a similar scale so they can be compared fairly. This helps the model learn without being overwhelmed by very large or very small numbers.

## Sequence Processing

### Temporal Splitting
**Technical**: Dividing data chronologically rather than randomly
**Simple Explanation**: Like testing a scout's predictions:
- Train them on 2022-2023 games
- Test them on 2024 games
This is more realistic than mixing future and past games together.

### Sequence Length
**Technical**: Fixed-length windows of historical games (32 for QB, 16 for defense)
**Simple Explanation**: Looking at:
- Last 32 games for QB performance
- Last 16 games for defensive performance
Like a scout focusing on recent history but with enough games to spot patterns.

### Feature Sets
**QB Features** (16 total):
1. Basic Stats:
   - Yards gained
   - Touchdowns
   - Pass attempts
   - Air yards
   - Yards after catch

2. Protection Metrics:
   - QB hits
   - Sacks

3. Pass Distribution:
   - Short passes
   - Deep passes
   - Left passes
   - Middle passes
   - Right passes

4. Efficiency Metrics:
   - Completion percentage
   - Yards per attempt
   - Sack rate
   - Deep pass rate

**Defense Features** (11 total):
1. Basic Stats:
   - Yards allowed
   - Touchdowns allowed
   - Pass attempts against
   - Air yards allowed
   - Yards after catch allowed

2. Pressure Stats:
   - QB hits
   - Sacks

3. Coverage Stats:
   - Short passes allowed
   - Deep passes allowed

4. Efficiency Metrics:
   - Completion percentage allowed
   - Sack rate

## Training Concepts

### Loss Functions
**Technical**: Mathematical measure of prediction error (MSE)
**Simple Explanation**: Like a score that shows how far off the predictions are. If predicting 300 passing yards and the actual was 250, the loss function measures how bad that miss was. Lower score = better predictions.

### Gradient Descent
**Technical**: Optimization algorithm that minimizes loss by adjusting weights
**Simple Explanation**: Like playing "hot and cold" - the model makes a prediction, sees how wrong it was, and adjusts to get "warmer". It keeps doing this until it can't get any better.

### Early Stopping
**Technical**: Technique to prevent overfitting by monitoring validation loss
**Simple Explanation**: Knowing when to stop practicing. Just like how over-practicing one specific game plan might make you worse against new opponents, we stop training when the model stops getting better at predicting new games.

### Batch Processing
**Technical**: Training on small subsets of data at a time
**Simple Explanation**: Like studying flashcards in small groups instead of trying to memorize the whole deck at once. It's more manageable and helps learn patterns better.

## Neural Network Layers

### Hidden Layers
**Technical**: Internal layers between input and output that learn representations of the data
**Simple Explanation**: Like the brain's internal processing steps between seeing something and making a decision.

**Our Model's Hidden Layers**:
```
Input → Hidden Layers → Output
[Raw Stats] → [64] → [32] → [5 predictions]

More detailed:
[QB+Def Features] → [LSTM Hidden(64)] → [Dense(64)] → [Dense(32)] → [5 predictions]
                          ↑                 ↑            ↑
                    Hidden Layer 1    Hidden Layer 2    Hidden Layer 3
```

**Think of it like a scout's thought process**:
1. Input Layer: Raw game data
   - "Mahomes threw for 300 yards last game"
   - "Bills defense allowed 200 passing yards per game"

2. First Hidden Layer (64 neurons):
   - Learns basic patterns
   - "Mahomes is on a hot streak"
   - "Bills struggle against mobile QBs"

3. Second Hidden Layer (32 neurons):
   - Combines patterns
   - "This matchup favors passing plays"
   - "Weather conditions affect deep throws"

4. Output Layer:
   - Final predictions
   - "Predict: 285 yards, 2 TDs"

**Why Hidden Layers Matter**:
- More layers = more complex patterns
- Like having multiple scouts:
  - Scout 1 watches the QB
  - Scout 2 watches the defense
  - Scout 3 combines their insights
  - Head Scout makes final prediction

**In Our Model**:
1. LSTM Hidden States (64 dimensions)
   - Remember important game patterns
   - Track player development
   - Notice trends

2. Dense Hidden Layers
   - Layer 1: 64 neurons
   - Layer 2: 32 neurons
   - Each helps refine predictions

**Real-World Analogy**:
Like a chef making a meal:
- Input: Raw ingredients (stats)
- Hidden Layer 1: Basic prep (chopping, seasoning)
- Hidden Layer 2: Combining ingredients
- Output: Final dish (predictions)

**Why Multiple Layers?**:
- Simple patterns combine into complex ones
- Each layer specializes in different aspects
- Helps model understand subtle relationships
- More layers = more learning capacity

**Trade-offs**:
- More layers = better learning potential
- But also:
  - Longer training time
  - More likely to overfit
  - Needs more data
  - Harder to train properly

## Model Components

### Sequence Processing
**Technical**: Handling ordered data while maintaining temporal relationships
**Simple Explanation**: Looking at games in order, like a scout watching a season unfold. Recent games matter more than games from months ago, and the order helps spot trends (like improving accuracy or declining performance).

### Exponential Weighting
**Technical**: w_t = exp(-λ(T-t)) giving more importance to recent events
**Simple Explanation**: Like your memory - yesterday's game is very clear, last week's is a bit fuzzy, and last month's is even hazier. We weight recent games more heavily than older ones.

### Dropout
**Technical**: Randomly disabling neurons during training
**Simple Explanation**: Like practicing with different players absent - it forces the team to not rely too much on any one player. This makes the model more robust and prevents it from becoming too dependent on specific patterns.

## Statistical Concepts

### Standard Scaling
**Technical**: Normalizing features to mean=0, std=1
**Simple Explanation**: Converting different stats to the same scale. Instead of comparing yards (200-400) directly with touchdowns (0-5), we convert them to comparable numbers.

### Train-Test Split
**Technical**: Dividing data into training (80%) and testing (20%) sets
**Simple Explanation**: Like practicing plays (training) vs using them in a real game (testing). We hold back some data to make sure our predictions work on games the model hasn't seen before.

## Evaluation Metrics

### Training Loss
**Technical**: Error measurement on training data using Mean Squared Error (MSE)
**Simple Explanation**: How well the model predicts games it has practiced on. Like a practice score.

**Understanding Loss Values**:
- For our QB predictions:
  - Loss ≈ 0.5: Very good (predictions off by ~0.7 standard deviations)
  - Loss ≈ 1.0: Decent (predictions off by ~1 standard deviation)
  - Loss ≈ 2.0: Poor (predictions off by ~1.4 standard deviations)

**Real-World Context**:
For passing yards (std ≈ 75 yards):
- Loss of 0.5 means predictions typically within ±53 yards
- Loss of 1.0 means predictions typically within ±75 yards
- Loss of 2.0 means predictions typically within ±106 yards

For touchdowns (std ≈ 1.2 TDs):
- Loss of 0.5 means predictions typically within ±0.8 TDs
- Loss of 1.0 means predictions typically within ±1.2 TDs
- Loss of 2.0 means predictions typically within ±1.7 TDs

**Example**:
If model predicts Mahomes for 300 yards, 2 TDs:
- Good Loss (0.5): Actual might be 250-350 yards, 1-3 TDs
- Decent Loss (1.0): Actual might be 225-375 yards, 1-4 TDs
- Poor Loss (2.0): Actual might be 200-400 yards, 0-4 TDs

### Validation Loss
**Technical**: Error measurement on unseen data
**Simple Explanation**: How well the model predicts new games it hasn't seen. Like game day performance.

**What to Look For**:
1. Training Loss: Should decrease over time
2. Validation Loss: Should track close to training loss
3. Gap between losses: Small gap is good, large gap means overfitting
4. Early Stopping: Stops when validation loss stops improving

**Current Model Performance**:
- Training Loss: ~0.90
- Validation Loss: ~1.02
- Interpretation: Model is doing okay but has room for improvement
- This means predictions are typically off by about one standard deviation

## Prediction Targets

### Core Statistics
What the model actually predicts:
1. **Passing Yards**: Total yards thrown
2. **Touchdowns**: Passing touchdowns thrown
3. **Interceptions**: Passes caught by the defense
4. **Completion Percentage**: Percent of passes completed
5. **Yards per Attempt**: Average yards gained per throw

## Real-World Example

Imagine you're predicting how Patrick Mahomes will perform against the Bills:
1. The model looks at:
   - Mahomes' last 16 games (weighted toward recent games)
   - Bills' defensive performances
   - Historical Mahomes vs Bills matchups
   - General Mahomes and Bills team patterns

2. It combines all this information to predict:
   - "Likely 285 passing yards"
   - "About 2.3 touchdowns"
   - "70% completion rate"
   etc.

3. The predictions are based on:
   - Recent form (last few games matter most)
   - Historical patterns
   - Specific matchup dynamics
   - Overall player and team tendencies