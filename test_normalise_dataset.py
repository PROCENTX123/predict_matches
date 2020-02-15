from normalise_dataset import normalize_dataset_values
from utils import read_file_local


def test_normalize_dataset_values():
    players = read_file_local('data_files/selected_players.csv')
    print(normalize_dataset_values(players[['gold_sell', 'gold_destroying_structure', 'gold_killing_roshan']].fillna(0)))

test_normalize_dataset_values()