#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Performs grid search for capacity parameter of svm rank as in "Mapping Bug Reports to Relevant Files:
A Ranking Model, a Fine-Grained Benchmark, and Feature Evaluation"

Requires "svm_rank_learn" in the same folder
Requires results of save_normalized_fold_dataframes.py

Converts data to svm rank format:
3 qid:1 1:0.53 2:0.12
2 qid:1 1:0.13 2:0.1
7 qid:2 1:0.87 2:0.12 
https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
"""
import json

import numpy as np
import pandas as pd
import subprocess
import sys
from sklearn.datasets import dump_svmlight_file

from metrics import calculate_metrics, calculate_metric_results


def main():
    file_prefix = sys.argv[1]

    process(file_prefix)


def load_fold_number(file_prefix):
    with open(file_prefix + '_fold_info', 'r') as f:
        fold_info = json.load(f)
        fold_number = fold_info['fold_number']
        print('fold number', fold_number)
        return fold_number


def process(file_prefix):
    fold_number = load_fold_number(file_prefix)

    fold_training = {}
    fold_testing = {}

    for k in range(fold_number + 1):
        fold_training[k] = pd.read_pickle(file_prefix + '_normalized_training_fold_' + str(k))
        fold_testing[k] = pd.read_pickle(file_prefix + '_normalized_testing_fold_' + str(k))
        print('fold_training', str(k), 'shape', fold_training[k].shape)
        print('fold_testing', str(k), 'shape', fold_testing[k].shape)

    grid_search_data_training = fold_training[0].copy()
    grid_search_data_testing = fold_testing[0].copy()

    print('grid search data training shape', grid_search_data_training.shape)
    print('grid search data testing shape', grid_search_data_testing.shape)
    print('grid search data used in fix equal 1.0',
          grid_search_data_training['used_in_fix'][(grid_search_data_training['used_in_fix'] == 1.0)].count())
    print('grid search data used in fix equal 0.0',
          grid_search_data_training['used_in_fix'][(grid_search_data_training['used_in_fix'] == 0.0)].count())
    grid_search_data_null_columns = grid_search_data_training.columns[grid_search_data_training.isnull().any()]
    print('grid search data null columns', grid_search_data_null_columns)
    # exit(0)
    k_range = range(1, 21)
    c = run_grid_search(grid_search_data_training, grid_search_data_testing, file_prefix, k_range)

    evaluate_algorithm(c, fold_training, fold_testing, fold_number, file_prefix, k_range)


def evaluate_algorithm(c, fold_training, fold_testing, fold_number, file_prefix, k_range):
    print('Using svm rank c', c)
    mean_accuracy_at_k = dict.fromkeys(k_range, 0)
    mean_mean_average_precision = 0.0
    mean_mean_reciprocal_rank = 0.0
    weights_at_fold = {}
    for i in range(fold_number):
        print('training on fold', i)
        current_training_file = save_svm_rank_data(fold_training[i], file_prefix + '_fold_training_' + str(i))
        model_file_name = file_prefix + '_fold_model_' + str(i)
        run_svm_rank(c, current_training_file, model_file_name)
        weights = read_weights(model_file_name, 19)
        weights_at_fold[i] = weights
        print('testing on fold', i + 1)
        testing_df = fold_testing[i + 1]
        accuracy_at_k, mean_average_precision, mean_reciprocal_rank = check_average_precision(testing_df.copy(),
                                                                                              weights, k_range)
        for k in k_range:
            mean_accuracy_at_k[k] += accuracy_at_k[k]
        mean_mean_average_precision += mean_average_precision
        mean_mean_reciprocal_rank += mean_reciprocal_rank
    print("Accuracy at k in k range", k_range)
    for k in k_range:
        mean_accuracy_at_k[k] = mean_accuracy_at_k[k] / fold_number
        print(mean_accuracy_at_k[k])
    mean_mean_average_precision = mean_mean_average_precision / fold_number
    print("Mean mean average prediction", mean_mean_average_precision)
    mean_mean_reciprocal_rank = mean_mean_reciprocal_rank / fold_number
    print("Mean mean reciprocal rank", mean_mean_reciprocal_rank)
    print("Evaluate on whole dataset")

    all_data = []
    for i in range(fold_number):
        df = apply_weights(fold_testing[i + 1].copy(), weights_at_fold[i])
        all_data.append(df)
    all_data_df = pd.concat(all_data)
    all_data_df.to_pickle(file_prefix+'_grid_search_19_results')

    # tie_breaking_on_df(all_data_df, break_on='f2')
    calculate_metric_results(all_data_df)


def run_grid_search(training, testing, file_prefix, k_range):
    print('training shape', training.shape)

    index_pairs = training.index.values
    bug_reports = set()
    for index_pair in index_pairs:
        bug_reports.add(index_pair[0])

    bug_reports = list(bug_reports)
    verification_split = int(len(bug_reports) * 0.6)
    training_indexes = bug_reports[0:verification_split]
    verification_indexes = bug_reports[verification_split:]
    print("bug reports for grid search", len(bug_reports))
    print("verification split", verification_split)
    print("training_indexes", training_indexes)
    print("verification indexes", verification_indexes)

    svm_training = training.loc[training_indexes].copy()
    svm_verification = testing.loc[verification_indexes].copy()
    print('svm training shape', svm_training.shape)
    print('svm verification shape', svm_verification.shape)

    data_filename = save_svm_rank_data(svm_training, file_prefix + '_grid_search_training')

    c_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10, 100, 300, 1000]
    # c_range = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9] + list(range(1, 1000, 1))
    max_mean_average_precision = 0
    selected_c = None
    for c in c_range:
        mean_average_precision = train_svm_for_grid_search(c, data_filename, file_prefix, svm_verification.copy(),
                                                           k_range)
        if mean_average_precision > max_mean_average_precision:
            selected_c = c
            max_mean_average_precision = mean_average_precision
    return selected_c


def train_svm_for_grid_search(c, data_filename, file_prefix, svm_verification, k_range):
    print('Training for c', c)
    run_svm_rank(c, data_filename, file_prefix + '_grid_search_model')
    weights = read_weights(file_prefix + '_grid_search_model', 19)
    accuracy_at_k, mean_average_precision, mean_reciprocal_rank = check_average_precision(svm_verification, weights,
                                                                                          k_range)
    return mean_average_precision


def check_average_precision(verification_df, weights, k_range):
    df = apply_weights(verification_df, weights)
    return calculate_metrics(df, k_range)


def apply_weights(df, weights):
    result = (df.drop('used_in_fix', axis=1) * weights).sum(axis=1)
    df['result'] = result
    return df


def run_svm_rank(c, data_file_name, model_file_name):
    cmd = ['./svm_rank_learn', '-c', str(c), data_file_name, model_file_name]
    subprocess.call(' '.join(cmd), shell=True)


def read_weights(model_path, expected_weights_number):
    model_file_lines = open_model(model_path)
    return retrieve_weights_from_model(model_file_lines, expected_weights_number)


def open_model(model_path):
    print('model file ', model_path)
    with open(model_path) as model_file:
        model_file_lines = model_file.readlines()
        return model_file_lines


def retrieve_weights_from_model(model_file_lines, expected_weights_number):
    weights_line = model_file_lines[-1]
    parts = weights_line.split(' ')
    weights_parts = parts[1:-1]
    weights_dict = {}
    for weight_part in weights_parts:
        index, weight = weight_part.split(':')
        weights_dict[int(index) - 1] = float(weight)
    weights = []
    for i in range(expected_weights_number):
        if i in weights_dict:
            weights.append(weights_dict[i])
        else:
            weights.append(0.0)
    return weights


def save_svm_rank_data(dataframe, name):
    X = dataframe[
        ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16',
         'f17', 'f18', 'f19']].values
    # X = dataframe[['f1', 'f2', 'f3', 'f4', 'f5', 'f6']].values
    y = np.squeeze(dataframe[['used_in_fix']].values)

    multi_index = dataframe.index
    q_id = multi_index.labels[0]
    file_name = name + '_svm_rank'
    dump_svmlight_file(X, y, file_name, zero_based=False, query_id=q_id)
    return file_name


if __name__ == '__main__':
    main()
