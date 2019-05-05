import json
import numpy as np
import pandas as pd
import sys
from source.new_feature import targets_names
import tsfresh
from tsfresh.feature_extraction import extract_features

# items_id = list(pd.read_csv('ai-academy-2019-skill-prediction-data-csv-v1/dota2_items.csv')['item_id'])
json_train = './sberbank_data/data_2019/dota2_skill_train.jsonlines'

items_df = pd.read_csv('./sberbank_data/data_2019/dota2_items.csv')
items_df.tail()
item_to_index_dict = dict(zip(items_df['item_id'], items_df.index))

targets_to_index_dict = dict( zip(targets_names, np.arange(len(targets_names))) )

hero_df = pd.read_csv('sberbank_data/data_2019/dota2_heroes.csv')
hero_id = hero_df['hero_id'].values
hero_name = hero_df['name'].values
hero_dict = dict(zip(hero_id, hero_name))

player_team_to_enemy_team = {'radiant' : 'dire', 'dire' : 'radiant'}
columns_gold = ['mean_plr_gold', 'var_plr_gold', 'min_plr_gold', 'max_plr_gold']
columns_gold += ['mean_tm_gold', 'var_tm_gold', 'min_tm_gold', 'max_tm_gold']
columns_gold += ['mean_enemy_gold', 'var_enemy_gold', 'min_enemy_gold', 'max_enemy_gold']
def get_gold_features(lines):
    """
    returns:
    --------
    - data:
        numpy array, shape (n, 4)
    """

    feats = []
    for row in lines:
        player_gold = np.diff(row['series']['player_gold'][18:])

        player_team = row['player_team']
        enemy_team = player_team_to_enemy_team[row['player_team']]

        teammates_gold = np.diff(row['series'][player_team + '_gold'][18:])
        enemies_gold = np.diff(row['series'][enemy_team + '_gold'][18:])

        feats.append([player_gold.mean(),
            player_gold.var(),
            player_gold.min(),
            player_gold.max(),

            teammates_gold.mean(),
            teammates_gold.var(),
            teammates_gold.min(),
            teammates_gold.max(),

            enemies_gold.mean(),
            enemies_gold.var(),
            enemies_gold.min(),
            enemies_gold.max(),
            ])

    return np.array(feats)


columns_items = ['item_%d' % x for x in range(233)]
def get_items_features(lines, n=None, placeholder=-1000):
    """
    parameters:
    -----------
    - lines
        лист уже прочитанных json строк

    - n
        длина списка lines (да, можно вычислить, но так быстрее)

    - placeholder
        чем заполнять если item не был куплен

    returns:
    --------
    - data
        numpy array, shape (n, 233)
    """
    if n is None:
        n = len(lines)

    items = np.ones((n, 233)) * placeholder
    for i, row in enumerate(lines):
        for item_log in row['item_purchase_log']:
            items[i, item_to_index_dict[item_log['item_id']]] = item_log['time']

    return items


columns_damage = targets_names
def get_damage_features(lines, n = None):
    """
    parameters:
    -----------
    - lines
        лист уже прочитанных json строк

    returns:
    --------
    - data
        numpy array, shape (n, 116)
    """
    if n is None:
        n = len(lines)

    damage = np.zeros( (n, len(targets_names)) )

    for i, row in enumerate(lines):
        damage_dict = row['damage_targets']
        for k, v in damage_dict.items():
            damage[i, targets_to_index_dict[k]] = v

    return damage


columns_lvl = ['lvl_%d' % x for x in range(2,26)]
def get_lvl_times(lines, n=None, placeholder=-1 ):
    """
    parameters:
    -----------
    - lines
        лист уже прочитанных json строк

    returns:
    --------
    - data
        numpy array, shape (n, 116)
    """
    if n is None:
        n = len(lines)
    levels_time = np.ones((n, 24)) * placeholder

    for i,elem in enumerate(lines):
        times = elem['level_up_times']
        levels_time[i, 0:len(times)] = times

    return levels_time


columns_abililies = ['abil_%d' % x for x in range(1,26)]
def get_abilities(lines, n=None, placeholder=-1):
    """
    parameters:
    -----------
    - lines
        лист уже прочитанных json строк

    returns:
    --------
    - data
        numpy array, shape (n, 116)
    """
    if n is None:
        n = len(lines)

    abil_id = np.ones((n, 25)) * placeholder

    for i,elem in enumerate(lines):
        abil_list = elem['ability_upgrades']



        abil_id[i, 0:len(abil_list)] = abil_list

    return abil_id

columns_items_value = ['items_value']
def get_items_value(lines,n=None):

    if n is None:
        n = len(lines)


    items_values = []
    for elem in lines:
        value = len(elem['item_purchase_log'])
        items_values.append(value)

    return np.array(items_values).reshape(n, 1)


columns_tar_value = ['targets_value']
def get_targets_value(lines, n=None):

    if n is None:
        n = len(lines)

    targets_value = []
    for elem in lines:
        value = len(elem['damage_targets'])
        targets_value.append(value)

    return np.array(targets_value).reshape(n, 1)


columns_friend_damage = ['friend_damage']
def get_friend_damage(lines, n=None):

    if n is None:
        n = len(lines)
        team_damage_lst = []

    for sample in lines:

        player_team = sample['player_team']
        friend_team = sample[player_team+'_heroes']
        hero_id = sample['hero_id']
        try:
            friend_team.remove(hero_id)
        except:
            print(hero_id, friend_team)
        team_damage = 0
        for ids in friend_team:
            try:
                team_damage += record['damage_targets'][hero_dict[ids]]
            except:
                pass

        team_damage_lst.append(team_damage)

        return np.array(team_damage).reshape(n, 1)

columns_self_damage = ['self_damage']
def get_self_damage(lines, n = None):
    if n is None:
        n = len(lines)
    self_damage_lst = []

    for sample in lines:
        player_team = sample['player_team']
        friend_team = sample[player_team+'_heroes']
        hero_id = sample['hero_id']
        self_damage = 0

        try:
            self_damage += record['damage_targets'][hero_dict[hero_id]]
        except:
            self_damage+=0
        self_damage_lst.append(self_damage)

    return np.array(self_damage_lst).reshape(n, 1)

columns_deviation =['friend_dev','enemy_dev']
def gold_deviation(lines, n=None):
    """
    Расчитывает отклонение по золоту.
    Return friend_dev , enemy_dev
    """

    data = []
    for json_line in lines:
        player_team = json_line['player_team']

        if player_team == 'dire':
            enemy_team = 'radiant'
        else:
            enemy_team ='dire'


            player_series = json_line['series']['player_gold'][18:]
            friend_series = json_line['series'][player_team+'_gold'][18:]
            enemy_series = json_line['series'][enemy_team+'_gold'][18:]

            mean_friend_team = np.subtract(friend_series, player_series) / 4
            mean_enemy_series = np.array(enemy_series) / 5

            friend_dev = np.mean(np.subtract(player_series, mean_friend_team))
            enemy_dev = np.mean(np.subtract(player_series, mean_enemy_series))

            data.append([friend_dev, enemy_dev])

    return np.array(data)


columns_items_freq = ['item_freq_%d' % x for x in range(233)]
def get_items_frequency(lines, n=None, placeholder=0):
    """
    parameters:
    -----------
    - lines
        лист уже прочитанных json строк

    - n
        длина списка lines (да, можно вычислить, но так быстрее)

    - placeholder
        чем заполнять если item не был куплен

    returns:
    --------
    - data
        numpy array, shape (n, 233)
    """
    if n is None:
        n = len(lines)

    items = np.ones((n, 233)) * placeholder
    for i, row in enumerate(lines):
        for item_log in row['item_purchase_log']:
            items[i, item_to_index_dict[item_log['item_id']]] += 1

    return items


def poly_trend(timeseries, degree=1):
    """
    Поиск прямой, которая максимально соответствует тренду TS
    Возвращает коэффициенты линейной функции вида f(x) = ax + b
    """
    X = np.arange(0, len(timeseries))
    Y = np.array(timeseries)
    z = np.polyfit(X, Y, degree)

    return z[0],z[1]


columns_trend = ['a_gold','a_level','a_friend_trend', 'a_enemy_trend']
def timeseries_trend(lines):
    """
    Считает коэфф прямой тренда уровня, личного золота
    и среднего золота на игрока по обоим командам
    Возвращает лист с коэффициентами a, b (функции вида y = ax+b)
    для каждого из TS
    """

    data = []
    for record in lines:
        player_team = record['player_team']

        if player_team == 'dire':
            enemy_team = 'radiant'
        else:
            enemy_team ='dire'

        player_gold = record['series']['player_gold'][18:]
        friend_series = record['series'][player_team+'_gold'][18:]
        enemy_series = record['series'][enemy_team+'_gold'][18:]

        player_level_times = record['level_up_times']

        a_gold, b_gold = poly_trend(player_gold)
        a_level, b_level = poly_trend(player_level_times)
        # mean_team_gold

        mean_friend_team = np.subtract(friend_series, player_gold) / 4
        mean_enemy_team = np.array(enemy_series) / 5

        a_friend_trend, b_friend_trend = poly_trend(mean_friend_team)
        a_enemy_trend, b_enemy_trend  = poly_trend(mean_enemy_team)

        data.append([a_gold, a_level, a_friend_trend, a_enemy_trend])

    return np.array(data)
