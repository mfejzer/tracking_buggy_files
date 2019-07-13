#!/usr/bin/env python3
# -*- coding: utf-8-unix -*-

"""Utilities to calculate evaluation metrics Accuracy@k, MAP and MRR
Usage: %(scriptName) <pickled dataframe file>

Handles data in pandas dataframes.
Each row in dataframe should correspond metrics of specific one file.
Dataframe should be indexed by tuple multiindex (first index is bug report, second is file sha)
'result' should contain rank calculated by ranking algorithm
'used_in_fix' should contain 0 or 1 if specific file was used in fix
"""

import numpy as np
import pandas as pd
import sys


def calculate_metrics(verification_df, k_range=range(1, 21)):
    average_precision_per_bug_report = []
    reciprocal_ranks = []
    # calculate per each query (bug report)
    accuracy_at_k = dict.fromkeys(k_range, 0)
    bug_report_number = 0
    for bug_report, bug_report_files_dataframe in verification_df.groupby(level=0, sort=False):
        min_fix_result = bug_report_files_dataframe[bug_report_files_dataframe['used_in_fix'] == 1.0]['result'].min()
        bug_report_files_dataframe2 = bug_report_files_dataframe[bug_report_files_dataframe["result"] >= min_fix_result]
        sorted_df = bug_report_files_dataframe2.sort_values(ascending=False, by=['result'])
        if sorted_df.shape[0] == 0:
            sorted_df = bug_report_files_dataframe.copy().sort_values(ascending=False, by=['result'])
            # print((bug_report_files_dataframe['used_in_fix'] == 1.0).sum())

        precision_at_k = []
        # precision per k in range
        tmp = sorted_df
        a = range(1, tmp.shape[0] + 1)
        tmp['position'] = pd.Series(a, index=tmp.index)

        large_k_p = tmp[(tmp['used_in_fix'] == 1.0)]['position'].tolist()
        unique_results = sorted_df['result'].unique().tolist()
        unique_results.sort()
        for fk in large_k_p:
            k = int(fk)
            k_largest = unique_results[-k:]

            largest_at_k = sorted_df[sorted_df['result'] >= min(k_largest)]
            real_fixes_at_k = (largest_at_k['used_in_fix'] == 1.0).sum()

            p = float(real_fixes_at_k) / float(k)
            precision_at_k.append(p)

        # average precision is sum of k precisions divided by K
        # K is set of positions of relevant documents in the ranked list
        average_precision = pd.Series(precision_at_k).mean()
        # average_precision = pd.Series(precision_at_k).sum() / float(large_k)
        average_precision_per_bug_report.append(average_precision)

        # accuracy
        for k in k_range:
            k_largest = unique_results[-k:]

            largest_at_k = sorted_df[sorted_df['result'] >= min(k_largest)]
            real_fixes_at_k = largest_at_k['used_in_fix'][(largest_at_k['used_in_fix'] == 1.0)].count()
            if real_fixes_at_k >= 1:
                accuracy_at_k[k] += 1

        # reciprocal rank
        indexes_of_fixes = np.flatnonzero(sorted_df['used_in_fix'] == 1.0)
        if indexes_of_fixes.size == 0:
            reciprocal_ranks.append(0.0)
        else:
            first_index = indexes_of_fixes[0]
            reciprocal_rank = 1.0 / (first_index + 1)
            reciprocal_ranks.append(reciprocal_rank)
        # bug number
        bug_report_number += 1

        del bug_report, bug_report_files_dataframe

    # print("average_precision_per_bug_report", average_precision_per_bug_report)
    mean_average_precision = pd.Series(average_precision_per_bug_report).mean()
    # print('mean average precision', mean_average_precision)
    mean_reciprocal_rank = pd.Series(reciprocal_ranks).mean()
    # print('mean reciprocal rank', mean_reciprocal_rank)
    for k in k_range:
        accuracy_at_k[k] = accuracy_at_k[k] / bug_report_number
        # print('accuracy for k', accuracy_at_k[k], k)
    return accuracy_at_k, mean_average_precision, mean_reciprocal_rank


def main():
    df_path = sys.argv[1]
    df = pd.read_pickle(df_path)
    print(calculate_metric_results(df))


def calculate_metric_results(df, k_range=range(1, 21)):
    all_data_accuracy_at_k, all_data_mean_average_precision, all_data_mean_reciprocal_rank = \
        calculate_metrics(df, k_range)
    return all_data_accuracy_at_k, all_data_mean_average_precision, all_data_mean_reciprocal_rank, k_range


def print_metrics(all_data_accuracy_at_k, all_data_mean_average_precision, all_data_mean_reciprocal_rank, k_range):
    print("All data accuracy at k in k range", k_range)
    for k in k_range:
        print(all_data_accuracy_at_k[k])
    print("All data mean average precision", all_data_mean_average_precision)
    print("All data mean reciprocal rank", all_data_mean_reciprocal_rank)


if __name__ == '__main__':
    main()
