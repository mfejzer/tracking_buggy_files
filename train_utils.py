#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Loads data from each fold for training and testing

Requires results of save_normalized_fold_dataframes.py
"""
from __future__ import print_function
import json

import pandas as pd
import sys

import sys

def eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)


def main():
    file_prefix = sys.argv[1]

    load_data_folds(file_prefix)


def load_fold_number(file_prefix):
    with open(file_prefix + '_fold_info', 'r') as f:
        fold_info = json.load(f)
        fold_number = fold_info['fold_number']
        eprint('fold number', fold_number)
        return fold_number


def load_data_folds(file_prefix):
    fold_number = load_fold_number(file_prefix)
    fold_training = {}
    fold_testing = {}
    for k in range(fold_number + 1):
        fold_training[k] = pd.read_pickle(file_prefix + '_normalized_training_fold_' + str(k))
        fold_testing[k] = pd.read_pickle(file_prefix + '_normalized_testing_fold_' + str(k))
        eprint('fold_training', str(k), 'shape', fold_training[k].shape)
        eprint('fold_testing', str(k), 'shape', fold_testing[k].shape)
    eprint("data loaded")
    return fold_number, fold_testing, fold_training


if __name__ == '__main__':
    main()
