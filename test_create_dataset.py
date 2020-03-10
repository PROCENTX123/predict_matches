from create_dataset import get_hero_pairs, get_kda, get_net_worth, get_hero_avg_duration, player_avg_stats, \
    create_team_feature_dataset, create_dataset_with_features
from utils import read_file_local


def test_get_hero_pairs():
    players = read_file_local('data_files/selected_players.csv')
    matches = read_file_local('data_files/match.csv')
    get_hero_pairs(players, matches)


def test_additional_features():
    players = read_file_local('data_files/selected_players.csv')
    duration_heroes = read_file_local('data_files/duration_heroes.csv')
    players[['deaths', 'kills', 'assists']].apply(get_kda, axis=1)
    players[['gold', 'gold_spent']].apply(get_net_worth, axis=1)
    players[['hero_id']].apply(lambda x: get_hero_avg_duration(x, duration_heroes), axis=1)


def test_player_avg_stats():
    players = read_file_local('data_files/selected_players.csv')
    matches = read_file_local('data_files/match.csv')
    feature_columns = ['unit_order_move_item',
                       'gold_sell',
                       'gold_destroying_structure',
                       'gold_killing_roshan',
                       'hero_damage', ]
    player_avg_stats(players, matches, feature_columns)


def test_create_team_feature_dataset():
    players = read_file_local('data_files/selected_players.csv')
    matches = read_file_local('data_files/match.csv')
    feature_columns = ['unit_order_move_item',
                       'gold_sell',
                       'gold_destroying_structure',
                       'gold_killing_roshan',
                       'hero_damage', ]
    avg_stats = player_avg_stats(players, matches, feature_columns)
    hero_pairs = get_hero_pairs(players, matches)
    create_team_feature_dataset(matches, players, avg_stats, hero_pairs)


# test_get_hero_pairs()
# test_additional_features()
# test_player_avg_stats()
# test_create_team_feature_dataset()

def test_full_dataset():
    players = read_file_local('data_files/selected_players.csv')
    duration_heroes = read_file_local('data_files/duration_heroes.csv')
    players["kda"] = players[['deaths', 'kills', 'assists']].apply(get_kda, axis=1)
    players["net_worth"] = players[['gold', 'gold_spent']].apply(get_net_worth, axis=1)
    players["duration_heroes"] = players[['hero_id']].apply(lambda x: get_hero_avg_duration(x, duration_heroes), axis=1)
    observed_features = ['gold_per_min', 'xp_per_min', 'denies', 'last_hits', 'hero_damage',
                         'hero_healing', 'tower_damage', 'kda', 'net_worth', 'duration_heroes',]
    create_dataset_with_features(players=players, matches=read_file_local('data_files/match.csv'),
                                 observed_features=observed_features,
                                 all_features=observed_features + ["hero_relative_strength"]
                                 ).to_csv("data_files/original_ds_not_normalized1.csv", index=False)


test_full_dataset()
