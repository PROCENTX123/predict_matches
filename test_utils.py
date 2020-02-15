from utils import read_file_local


def test_read_file():
    players = read_file_local('data_files/selected_players.csv')
    matches = read_file_local('data_files/match.csv')

test_read_file()
