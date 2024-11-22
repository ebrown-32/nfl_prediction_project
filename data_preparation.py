import nfl_data_py as nfl
import pandas as pd
import numpy as np

# Load and process NFL data
years = [2022, 2023, 2024]

print("Loading play by play data...")
play_by_play = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)

# Filter for passing plays
pass_plays = play_by_play[play_by_play['pass_attempt'] == 1]

print("\nAggregating game-level statistics...")
game_data = pass_plays.groupby(['game_id', 'passer_player_name', 'posteam', 'defteam', 'week', 'season']).agg({
    # Basic passing stats
    'pass_attempt': 'sum',
    'complete_pass': 'sum',
    'incomplete_pass': 'sum',
    'yards_gained': 'sum',
    'pass_touchdown': 'sum',
    'interception': 'sum',
    
    # Pass depth and route metrics
    'pass_length': lambda x: x.value_counts().to_dict(),
    'pass_location': lambda x: x.value_counts().to_dict(),
    'air_yards': 'mean',
    'yards_after_catch': 'mean',
    'qb_epa': 'mean',
    
    # Protection metrics
    'qb_hit': 'sum',
    'sack': 'sum',
    
}).reset_index()

# Helper functions to extract counts
def extract_pass_depths(row):
    depths = row['pass_length'] if isinstance(row['pass_length'], dict) else {}
    return pd.Series({
        'short_passes': depths.get('short', 0),
        'deep_passes': depths.get('deep', 0)
    })

def extract_pass_locations(row):
    locations = row['pass_location'] if isinstance(row['pass_location'], dict) else {}
    return pd.Series({
        'left_passes': locations.get('left', 0),
        'middle_passes': locations.get('middle', 0),
        'right_passes': locations.get('right', 0)
    })

# Add pass depth and location columns
game_data = pd.concat([
    game_data,
    game_data.apply(extract_pass_depths, axis=1),
    game_data.apply(extract_pass_locations, axis=1)
], axis=1)

# Drop the original dictionary columns
game_data.drop(['pass_length', 'pass_location'], axis=1, inplace=True)

# Calculate derived metrics
game_data['completion_percentage'] = (game_data['complete_pass'] / game_data['pass_attempt'] * 100).round(1)
game_data['yards_per_attempt'] = (game_data['yards_gained'] / game_data['pass_attempt']).round(1)
game_data['sack_rate'] = (game_data['sack'] / (game_data['pass_attempt'] + game_data['sack']) * 100).round(1)
game_data['deep_pass_rate'] = (game_data['deep_passes'] / game_data['pass_attempt'] * 100).round(1)

# Clean up and filter data
game_data = game_data.dropna(subset=['passer_player_name', 'pass_attempt'])
game_data = game_data[game_data['pass_attempt'] >= 10]  # Min attempts filter

print("\nSaving processed data...")
game_data.to_csv("game_data.csv", index=False)

# Print summary statistics
print("\nDataset Summary:")
print(f"Number of games: {len(game_data)}")
print(f"Number of unique quarterbacks: {game_data['passer_player_name'].nunique()}")
print(f"Number of unique teams: {game_data['defteam'].nunique()}")
print("\nPass depth distribution:")
print(f"Average deep pass rate: {game_data['deep_pass_rate'].mean():.1f}%")
print("\nPass location distribution:")
print(f"Left: {(game_data['left_passes'] / game_data['pass_attempt']).mean()*100:.1f}%")
print(f"Middle: {(game_data['middle_passes'] / game_data['pass_attempt']).mean()*100:.1f}%")
print(f"Right: {(game_data['right_passes'] / game_data['pass_attempt']).mean()*100:.1f}%")