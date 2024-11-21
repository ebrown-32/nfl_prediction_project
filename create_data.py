import nfl_data_py as nfl
import pandas as pd

# Load and process NFL data
years = [2022, 2023, 2024]

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

game_data.to_csv('game_data.csv', index=False)