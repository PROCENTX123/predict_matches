from collections import defaultdict
import itertools
import pandas as pd
import numpy as np

pairs_count = 2


def get_combinations_set(team_ids):
    return set([tuple(sorted(list(c))) for c in list(itertools.combinations(team_ids, pairs_count))])


def get_hero_pair_strength(hero_pair):
    return hero_pair["wins"] / (hero_pair["wins"] + hero_pair["losses"])


def get_hero_pairs(players, matches):
    hero_pairs = defaultdict(lambda: {"wins": 0, "losses": 0})
    for name, group in players.groupby('match_id'):
        match_id = name
        radiant_ids = group[group["player_slot"] < 100]["hero_id"]
        dire_ids = group[group["player_slot"] > 99]["hero_id"]
        radiant_win = matches[matches["match_id"] == match_id]["radiant_win"].tolist()[0]
        for pair in get_combinations_set(radiant_ids):
            if radiant_win:
                hero_pairs[pair]["wins"] += 1
            else:
                hero_pairs[pair]["losses"] += 1

        for pair in get_combinations_set(dire_ids):
            if not radiant_win:
                hero_pairs[pair]["wins"] += 1
            else:
                hero_pairs[pair]["losses"] += 1
    print(hero_pairs)
    return hero_pairs


# for apply(
def get_kda(players):
    if players['deaths'] > 0:
        return players['kills'] + players['assists'] / players['deaths']
    else:
        return players['kills'] + players['assists']


def get_net_worth(players):
    return players["gold"] + players["gold_spent"]


def get_hero_avg_duration(players, duration_heroes):
    hero_id = players['hero_id']
    res = duration_heroes.query('hero_id == {}'.format(hero_id))['Duration_heroes'].tolist()
    if not res:
        print(hero_id)
        raise
    return res[0]


def get_players_by_match_id(players_dataset, match_id):
    return players_dataset[players_dataset["match_id"] == match_id]


def get_features_by_player(player_id, players_dataset, feature_columns):
    return players_dataset[players_dataset["account_id"] == player_id][feature_columns]


def is_team_1_winner(match_df):
    return match_df['radiant_win']


def average_stats_for_player(match_ids, player_id_df, features):
    stats_df = player_id_df[player_id_df["match_id"].isin(match_ids)][features + ["account_id"]]
    return stats_df.groupby(['account_id']).mean()


def player_avg_stats(matches, players, feature_columns):
    matches_player_avg_stats = []
    for m_id in matches["match_id"]:
        if m_id % 1000 == 0:
            print(m_id)
        players_mid_df = players[players["match_id"] == m_id]
        for player_id in players_mid_df["account_id"]:
            if player_id != 0:
                previous_matches = matches[matches["match_id"] < m_id]["match_id"].tolist()
                current_player_id_df = players[players["account_id"] == player_id]
                avg_stats = average_stats_for_player(previous_matches, current_player_id_df, feature_columns)
                avg_stats["match_id"] = m_id
                matches_player_avg_stats.append(avg_stats)
    avg_stats = pd.concat(matches_player_avg_stats).reset_index()
    return avg_stats.fillna(0)


def create_team_feature_dataset(matches, players, avg_stats, hero_pairs):
    feature_dataset_by_team = []

    for _, _match in matches.iterrows():
        match_id = _match["match_id"]
        if match_id % 1000 == 0:
            print(match_id)
        players_in_match = get_players_by_match_id(players, match_id)

        team_1 = players_in_match[players_in_match["player_slot"] < 100]["account_id"].tolist()
        team_2 = players_in_match[players_in_match["player_slot"] >= 100]["account_id"].tolist()
        # filter 0s
        team_1_heroes = players_in_match[players_in_match["player_slot"] < 100]["hero_id"].tolist()
        team_2_heroes = players_in_match[players_in_match["player_slot"] >= 100]["hero_id"].tolist()

        team_1 = [_id for _id in team_1 if _id != 0]
        team_2 = [_id for _id in team_2 if _id != 0]

        if team_1 and team_2:
            players_avg_data_mid = avg_stats[avg_stats["match_id"] == match_id]
            team_1_data = players_avg_data_mid[players_avg_data_mid["account_id"].isin(team_1)]
            team_2_data = players_avg_data_mid[players_avg_data_mid["account_id"].isin(team_2)]
            if not team_1_data.empty and not team_2_data.empty:
                y = is_team_1_winner(_match)

                team_1_data.drop(columns=['match_id', 'account_id'], inplace=True)
                team_2_data.drop(columns=['match_id', 'account_id'], inplace=True)

                team_1_rel_str = np.mean(
                    [get_hero_pair_strength(hero_pairs[v]) for v in list(get_combinations_set(team_1_heroes))])
                team_2_rel_str = np.mean(
                    [get_hero_pair_strength(hero_pairs[v]) for v in list(get_combinations_set(team_2_heroes))])

                if np.isnan(team_1_rel_str):
                    team_1_rel_str = 0.5
                if np.isnan(team_2_rel_str):
                    team_2_rel_str = 0.5

                team_1_features = team_1_data.mean()
                team_1_features['Y'] = y

                team_1_features["hero_relative_strength"] = team_1_rel_str
                team_2_features = team_2_data.mean()
                team_2_features['Y'] = not y
                team_2_features["hero_relative_strength"] = team_2_rel_str

                feature_dataset_by_team.append(team_1_features)
                feature_dataset_by_team.append(team_2_features)

    feature_dataset_by_team = pd.DataFrame(feature_dataset_by_team)
    return feature_dataset_by_team.fillna(0)


def create_dataset_with_features(players, matches, feature_list):
    avg_stats = player_avg_stats(matches, players, feature_list)
    hero_pairs = get_hero_pairs(players, matches)
    return create_team_feature_dataset(matches, players, avg_stats, hero_pairs)

