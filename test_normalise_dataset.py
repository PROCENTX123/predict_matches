from normalise_dataset import normalize_dataset_values
from utils import read_file_local


def test_normalize_dataset_values():
    players = read_file_local('data_files/original_ds_not_normalized.csv')
    normalize_dataset_values(players.fillna(0)).to_csv("data_files/original_ds_normalized.csv", index=False)

test_normalize_dataset_values()