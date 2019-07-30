#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <feature_files_prefix>

Normalizes data from feature files, prepares as pandas dataframe per each fold, and saves those via pickle
Saves number of folds to '<feature_files_prefix>_fold_info' file

Requires results of calculate_vectorized_features.py
"""
import json

import pandas as pd
import numpy as np
import sys
from collections import defaultdict
from operator import itemgetter
from scipy import sparse
from tqdm import tqdm

from date_utils import convert_commit_date

feature_columns = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
]

def main():
    bug_report_file_path = sys.argv[1]
    file_prefix = sys.argv[2]

    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        process(bug_reports, file_prefix)


def process(bug_reports, file_prefix):
    sorted_ids = sort_bug_reports_by_db_id(bug_reports)
    # print(sorted_commits[0:5])
    #
    # exit()
    fold_training_data = defaultdict(list)
    fold_training_keys = defaultdict(list)

    fold_testing_data = defaultdict(list)
    fold_testing_keys = defaultdict(list)

    fold_size = 500

    fold_number = len(sorted_ids) // fold_size

    number_of_irrelevant_files = 200

    fold_index = 0
    for index, (bug_report_id, date) in enumerate(tqdm(sorted_ids)):
        features = load_features(file_prefix, bug_report_id)
        filenames = load_filenames(file_prefix, bug_report_id)
        df = pd.DataFrame(features.todense(), index=filenames)
        df.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15',
                      'f16', 'f17', 'f18', 'f19', 'used_in_fix']

        relevant = df[(df['used_in_fix'] == 1)]
        irrelevant = df[(df['used_in_fix'] == 0)].nlargest(number_of_irrelevant_files, 'f2')

        if relevant.shape[0] > 0:
            current_fold = fold_index // fold_size
            fold_index += 1

            training = pd.concat([relevant, irrelevant])

            fold_training_data[current_fold].append(training)
            fold_training_keys[current_fold].append(bug_report_id)

            fold_testing_data[current_fold].append(df)
            fold_testing_keys[current_fold].append(bug_report_id)

    fold_training = {}
    for fold_key, training_dataframes in fold_training_data.items():
        training_keys = fold_training_keys[fold_key]
        training_dataframe = pd.concat(training_dataframes, keys=training_keys)
        fold_training[fold_key] = training_dataframe

    min_dict = {}
    max_dict = {}

    fold_testing = {}
    for fold_key, testing_dataframes in fold_testing_data.items():
        testing_keys = fold_testing_keys[fold_key]
        testing_dataframe = pd.concat(testing_dataframes, keys=testing_keys)
        fold_testing[fold_key] = testing_dataframe

        mm_df = testing_dataframe.drop('used_in_fix', axis=1)
        min_dict[fold_key] = pd.DataFrame(mm_df.min()).transpose()
        max_dict[fold_key] = pd.DataFrame(mm_df.max()).transpose()

    print('max_dict', max_dict)
    print('min_dict', min_dict)

    save_normalized_data(file_prefix, fold_number, fold_testing, fold_training, max_dict, min_dict)


def save_normalized_data(file_prefix, fold_number, fold_testing, fold_training, max_dict, min_dict):
    # print('fold number', fold_number)
    # print(fold_training.keys())
    # print(fold_testing.keys())
    # exit(0)
    for current_fold_number in range(fold_number + 2):
        if current_fold_number == 0:
            k = 1
            min_df = min_dict[k]
            max_df = max_dict[k]
            df = fold_training[k]
            normalized_df = prepare_normalized_df(df, max_df, min_df)
            normalized_df.to_pickle(file_prefix + '_normalized_training_fold_' + str(current_fold_number))
            k = 0
            min_df = min_dict[k]
            max_df = max_dict[k]
            df = fold_testing[k]
            normalized_df = prepare_normalized_df(df, max_df, min_df)
            normalized_df.to_pickle(file_prefix + '_normalized_testing_fold_' + str(current_fold_number))
        else:
            k = current_fold_number - 1
            min_df = min_dict[k]
            max_df = max_dict[k]
            df = fold_training[k]
            normalized_df = prepare_normalized_df(df, max_df, min_df)
            normalized_df.to_pickle(file_prefix + '_normalized_training_fold_' + str(current_fold_number))
            print(normalized_df.max())
            print(normalized_df[normalized_df > 1.0].count())

            if k == 0:
                i = k
            else:
                i = k - 1
            min_df = min_dict[i]
            max_df = max_dict[i]
            df = fold_testing[k]
            normalized_df = prepare_normalized_df(df, max_df, min_df)
            normalized_df.to_pickle(file_prefix + '_normalized_testing_fold_' + str(current_fold_number))
            print("max", max_df.max())
            print("min", min_df.min())
            print('fold_testing', k)
            print(normalized_df.max())
            print(normalized_df[normalized_df > 1.0].count())

    # for k, df in fold_training.items():
    #     df.to_pickle(file_prefix + '_training_fold_' + str(k))
    #     min_df = min_dict[k]
    #     max_df = max_dict[k]
    #
    #     normalized_df = save_normalized_df(df, file_prefix, k, max_df, min_df)
    #
    #     print('fold_training', k)
    #     print(normalized_df.max())
    # for k, df in fold_testing.items():
    #     df.to_pickle(file_prefix + '_testing_fold_' + str(k))
    #     if k == 0:
    #         i = k
    #     else:
    #         i = k - 1
    #     min_df = min_dict[i]
    #     max_df = max_dict[i]
    #
    #     normalized_df = save_normalized_df(df, file_prefix, k, max_df, min_df)
    #
    #     print("max", max_df.max())
    #     print("min", min_df.min())
    #     print('fold_testing', k)
    #     print(normalized_df.max())
    #     print(normalized_df[normalized_df > 1.0].count())

    info = {'fold_number': fold_number + 1}
    print(info)
    with open(file_prefix + '_fold_info', 'w') as info_file:
        json.dump(info, info_file)


def prepare_normalized_df(df, max_df, min_df):
    normalized_df = (df.drop('used_in_fix', axis=1) - min_df.min()) / (max_df.max() - min_df.min())
    for column in feature_columns:
        values = np.array(normalized_df[column].values.tolist())
        normalized_df[column] = np.where(values > 1.0, 1.0, values).tolist()
    normalized_df['used_in_fix'] = df['used_in_fix']
    normalized_df = normalized_df.fillna(0.0)
    return normalized_df


def load_features(file_prefix, bug_report_id):
    file_path = file_prefix + '_' + bug_report_id + '_features.npz'
    features_data = sparse.load_npz(file_path).tocsr()
    return features_data


def load_filenames(file_prefix, bug_report_id):
    file_path = file_prefix + '_' + bug_report_id + '_files'
    with open(file_path, 'r') as f:
        files_list = json.load(f)
        return files_list


def sort_bug_reports_by_db_id(bug_reports):
    ids = []
    for index, id in enumerate(bug_reports):
        timestamp = bug_reports[id]['bug_report']['timestamp']
        ids.append((id, timestamp))

    sorted_ids = sorted(ids, key=itemgetter(1))
    return sorted_ids


if __name__ == '__main__':
    main()
