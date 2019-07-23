#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Requires results of save_normalized_fold_dataframes.py
"""

import os

import sys
import shutil

import numpy as np
from skopt import *
from train_utils import load_data_folds, eprint

feature_columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
                   'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']


def fix_index(fold_testing):
    max_level_0 = 0
    max_level_1 = 0
    for fold in fold_testing:
        level_0_uniq = (fold_testing[fold].index.get_level_values(0).unique().shape[0])
        if level_0_uniq > max_level_0:
            max_level_0 = level_0_uniq

        level_1_uniq = (fold_testing[fold].index.get_level_values(1).unique().shape[0])
        if level_1_uniq > max_level_1:
            max_level_1 = level_1_uniq

        fold_testing[fold].index.set_levels(range(fold * max_level_0, (fold + 1) * max_level_0), level=0, inplace=True)
        fold_testing[fold].index.set_levels(range(fold * max_level_1, (fold + 1) * max_level_1), level=1, inplace=True)

        fold_testing[fold] = fold_testing[fold].astype(np.float32, copy=False)


def main():
    file_prefix = sys.argv[1]
    cwd = os.getcwd()
    folder = cwd+'/joblib_memmap_' + file_prefix
    print(folder)

    try:
        shutil.rmtree(folder)
    except:
        eprint('Could not clean-up automatically.')

    fold_number, fold_testing, fold_training = load_data_folds(file_prefix)

    eprint(fold_testing[0].info(memory_usage='deep'))
    fix_index(fold_testing)
    eprint(fold_testing[0].info(memory_usage='deep'))
    fix_index(fold_training)

    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

    data_filename_memmap = os.path.join(folder, 'data_memmap')
    dump([fold_number, fold_testing, fold_training], data_filename_memmap)


if __name__ == "__main__":
    main()
