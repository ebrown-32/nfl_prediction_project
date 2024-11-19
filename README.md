Author: ebrown

Requires play-by-play dataset:
Docs: https://pypi.org/project/nfl-data-py/

import nfl_data_py as nfl

play_by_play = nfl.import_pbp_data(years, downcast=True, cache=False, alt_path=None)

play_by_play.to_csv("pbp_data.csv") 

