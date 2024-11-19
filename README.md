Author: ebrown

Requires play-by-play dataset:
Docs: https://pypi.org/project/nfl-data-py/

import nfl_data_py as nfl

play_by_play = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)

play_by_play.to_csv("pbp_data.csv") 


### How It Works
1. **Model Training**:

   - The model is trained on a dataset that includes all games played by all quarterbacks. This dataset contains various features (like pass attempts, air yards, yards after catch, etc.) and the target variable (passing yards).
   
   - The model learns patterns from this comprehensive dataset, understanding how different features correlate with passing yards.

2. **Prediction Process**:

   - When you want to predict the passing yards for a specific QB (e.g., P.Mahomes) against a specific opponent (e.g., BAL), the function retrieves all historical games played by that QB.
   
   - For each of those games, it creates a hypothetical scenario where the QB plays against the new opponent. This includes:
    
     - Setting the opponent's defensive stats (like yards allowed per game, TDs allowed, etc.) based on the new opponent.
     
     - Keeping the QB's performance metrics from each historical game intact.


3. **Making Predictions**:

   - The model then predicts passing yards for each of these historical game patterns against the new opponent.
   
   - This results in multiple predictions (one for each historical game), which can be averaged or analyzed to provide a final prediction.
