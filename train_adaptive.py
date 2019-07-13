#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <feature_files_prefix>

Requires results of save_normalized_fold_dataframes.py
"""

import json
import gc
import inspect
import os
import shutil
import sys
from functools import partial, update_wrapper
from itertools import product

import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed
from scipy.optimize import Bounds, minimize
from sklearn.decomposition import *
from sklearn.ensemble import *
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.svm import *
from skopt import *
from collections import defaultdict
from metrics import get_no_tie_on_df
from train_utils import eprint, load_data_folds
from sklearn.manifold import TSNE

from sklearn.cluster import DBSCAN, Birch, MeanShift, KMeans
from sklearn.mixture import GaussianMixture
from timeit import default_timer
import time
from sklearn.utils import safe_mask
from scipy.stats import kruskal, ttest_ind, levene, mannwhitneyu, ranksums

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
    file_prefix = sys.argv[1]

    cwd = os.getcwd()
    folder = cwd+'/joblib_memmap_' + file_prefix

    data_filename_memmap = os.path.join(folder, "data_memmap")
    fold_number, fold_testing, fold_training = load(data_filename_memmap, mmap_mode="r")

    models = [Adaptive_Process()]
    results = []
    for m in models:
        results.append(process(m, fold_number, fold_testing, fold_training, file_prefix))

    results = [r for r in results if r is not None]
    eprint('Results')
    print(results)


def _process(ptemplate, fold_training, fold_testing):
    clf = ptemplate.train(fold_training)
    result = ptemplate.predict(clf, fold_testing)
    return result


def process(ptemplate, fold_number, fold_testing, fold_training, file_prefix, tie_break="f1"):
    results_list = []

    for i in range(fold_number):
        r = _process(ptemplate, fold_training[i], fold_testing[i + 1])
        if r is None:
            del ptemplate
            gc.collect()
            return None

        min_fix_result = r[r["used_in_fix"] == 1.0]["result"].min()
        minimal_reasonable_set = r[r["result"] >= min_fix_result].copy()
        del r
        results_list.append(minimal_reasonable_set)

    training_time_list = ptemplate.training_time_list.copy()
    prescoring_log = ptemplate.prescoring_log.copy()
    regression_log = ptemplate.regression_log.copy()
    best_prescoring_log = ptemplate.best_prescoring_log.copy()
    best_regression_log = ptemplate.best_regression_log.copy()

    all_results_df = pd.concat(results_list)
    all_results_df.reset_index(level=1, drop=True, inplace=True)

    eprint(training_time_list)
    time_sum = sum([time for time, bug_reports_number, file_number in training_time_list])
    bug_reports_number_sum = sum([bug_reports_number for time, bug_reports_number, file_number in training_time_list])
    file_number_sum = sum([file_number for time, bug_reports_number, file_number in training_time_list])

    eprint("time_sum", time_sum)
    eprint("bug_reports_number_sum", bug_reports_number_sum)
    eprint("file_number_sum", file_number_sum)

    mean_time_bug_report_training = time_sum / bug_reports_number_sum
    mean_time_file_training = time_sum / file_number_sum

    eprint("mean_time_bug_report_training", mean_time_bug_report_training)
    eprint("mean_time_file_training", mean_time_file_training)

    training_time = {'time_sum': time_sum,
                     'bug_reports_number_sum': bug_reports_number_sum,
                     'file_number_sum': file_number_sum,
                     'mean_time_bug_report_training': mean_time_bug_report_training,
                     'mean_time_file_training': mean_time_file_training}
    eprint(training_time)

    results_timestamp = time.strftime("%Y%m%d%H%M%S")
    with open(file_prefix + '_' + ptemplate.name + '_training_time_'+results_timestamp, 'w') as time_file:
        json.dump(training_time, time_file)
    with open(file_prefix + '_' + ptemplate.name + '_prescoring_log_'+results_timestamp, 'w') as prescoring_log_file:
        json.dump(prescoring_log, prescoring_log_file)
    with open(file_prefix + '_' + ptemplate.name + '_regression_log_'+results_timestamp, 'w') as regression_log_file:
        json.dump(regression_log, regression_log_file)
    with open(file_prefix + '_' + ptemplate.name + '_best_prescoring_log_'+results_timestamp, 'w') as best_prescoring_log_file:
        json.dump(best_prescoring_log, best_prescoring_log_file)
    with open(file_prefix + '_' + ptemplate.name + '_best_regression_log_'+results_timestamp, 'w') as best_regression_log_file:
        json.dump(best_regression_log, best_regression_log_file)
    # if tie_break is not None or False:
    #     rank_by = all_results_df[tie_break]
    #     all_results_df['result'] = all_results_df['result'].rank(
    #         method='dense') + (rank_by / rank_by.sum())
    unique_results = all_results_df["result"].unique()
    all_results = all_results_df["result"]

    try:
        return {
            "name": ptemplate.name,
            "tie_break": tie_break,
            "ties": (unique_results.shape[0]) / all_results.shape[0],
            "results": get_no_tie_on_df(all_results_df),
        }
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        eprint(exc_type, fname, exc_tb.tb_lineno)
        eprint(ptemplate.name, e)
        return None


def _weights_normalize(weights):
    weights_sum = weights.sum()
    if weights_sum > 0:
        weights /= weights_sum

    return weights


def weights_chi2(df, columns):
    weights = chi2(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def weights_mutual_info_classif(df, columns):
    weights = mutual_info_classif(
        df[columns], df["used_in_fix"], discrete_features=False
    )
    weights = weights

    return _weights_normalize(weights)


def weights_FastICA(df, columns):
    m = FastICA(n_components=1)
    m.fit(df[columns])
    weights = m.components_[0]

    return _weights_normalize(weights)


def weights_variance(df, columns):
    fs = VarianceThreshold()
    fs.fit(df[columns])
    weights = fs.variances_
    weights[weights < 0] = 0

    return _weights_normalize(weights)


def weights_const(df, columns):
    return np.ones(df[columns].shape[1]) * 0.5


def weights_ExtraTreesClassifier(df, columns):
    tree = ExtraTreesClassifier(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_SGDClassifier(df, columns):
    tree = SGDClassifier(class_weight="balanced", shuffle=False)

    Y = df['used_in_fix']
    in_mask = (Y == 1)

    r_mask = (Y == 1)
    r_mask[:2*sum(in_mask)]=True

    tree.fit(df[r_mask][columns], df[r_mask]["used_in_fix"])
    weights = np.asanyarray(tree.coef_[0])
    weights += np.abs(np.amin(weights))

    return weights


def weights_GradientBoostingClassifier(df, columns):
    tree = GradientBoostingRegressor(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_AdaBoostClassifier(df, columns):
    tree = AdaBoostClassifier(n_estimators=100)
    tree.fit(df[columns], df["used_in_fix"])
    weights = tree.feature_importances_

    return _weights_normalize(weights)


def weights_kruskal_classif(df, columns):
    weights = kruskal_classif(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def kruskal_classif(X, y):
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = kruskal(*args)
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_ttest_ind_classif(df, columns):
    weights = ttest_ind_classif(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def ttest_ind_classif(X, y):
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = ttest_ind(*args, equal_var=False)
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_levene_mean(df, columns):
    weights = levene_mean(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def levene_mean(X, y):
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = levene(args[0], args[1], center='mean')
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_levene_median(df, columns):
    weights = levene_median(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def levene_median(X, y):
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = levene(args[0], args[1], center='median')
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_levene_trimmed(df, columns):
    weights = levene_trimmed(df[columns], df["used_in_fix"])
    weights = weights[0]

    return _weights_normalize(weights)


def weights_levene_mean_p_value(df, columns):
    weights = levene_mean(df[columns], df["used_in_fix"])
    weights = 1 - weights[1]

    return weights


def mannwhitneyu_p(X, y):
    ret_k = []
    ret_p = []
    args = [0,0]

    for column in X:
        args[0] = X[safe_mask(X, y == 0)][column]
        args[1] = X[safe_mask(X, y == 1)][column]
        #r = mannwhitneyu(args[0], args[1], alternative="less")
        r = ranksums(args[1], args[0])
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_mannwhitneyu_p(df, columns):
    weights = mannwhitneyu_p(df[columns], df["used_in_fix"])

    weights = weights[0]

    return _weights_normalize(weights)
    weights = weights[0]

    return weights


def weights_levene_mean_f_selection_k(df, columns, k):
    weights = levene_mean(df[columns], df["used_in_fix"])
    weights = weights[0]
    idx = np.argpartition(weights, -k)[-k:]
    weights = np.zeros(weights.shape)
    weights[idx] = 1

    return _weights_normalize(weights)


def weights_levene_median_f_selection_k(df, columns, k):
    weights = levene_median(df[columns], df["used_in_fix"])
    weights = weights[0]
    idx = np.argpartition(weights, -k)[-k:]
    weights = np.zeros(weights.shape)
    weights[idx] = 1

    return _weights_normalize(weights)


def levene_trimmed(X, y):
    ret_k = []
    ret_p = []

    for column in X:
        args = [X[safe_mask(X, y == k)][column] for k in np.unique(y)]
        r = levene(args[0], args[1], center='trimmed')
        ret_k.append(abs(r[0]))
        ret_p.append(r[1])
    return np.asanyarray(ret_k), np.asanyarray(ret_p)


def weights_var(df, columns):
    weights_var = np.var(df[df["used_in_fix"]==1][columns], axis=0)
    weights_var1 = np.var(df[df["used_in_fix"]==0][columns], axis=0)

    return weights_var / weights_var1


def weights_mean_var(df, columns):
    weights_var = np.var(df[df["used_in_fix"]==1][columns], axis=0)
    weights_mean = np.mean(df[df["used_in_fix"]==1][columns], axis=0)
    weights_var1 = np.var(df[df["used_in_fix"]==0][columns], axis=0)
    weights_var1_mean = np.mean(df[df["used_in_fix"]==0][columns], axis=0)

    return (weights_var / weights_mean) / (weights_var1 / weights_var1_mean)


def weights_median_absolute_deviation(df, columns):
    weights_median = np.median(df[df["used_in_fix"]==1][columns], axis=0)
    weights_mad = np.median(np.abs(df[df["used_in_fix"]==1][columns] - weights_median), axis=0)
    # weights_median1 = np.median(df[df["used_in_fix"]==0][columns], axis=0)
    # weights_mad1 = np.median(np.abs(df[df["used_in_fix"]==0][columns] - weights_median1, axis=0), axis=0)
    return weights_mad # / weights_mad1


def weights_maximum_absolute_deviation(df, columns):
    weights_max = np.max(df[df["used_in_fix"]==1][columns], axis=0)
    weights_mad = np.mean(np.abs(df[df["used_in_fix"]==1][columns] - weights_max), axis=0)
    # weights_median1 = np.median(df[df["used_in_fix"]==0][columns], axis=0)
    # weights_mad1 = np.mean(np.abs(df[df["used_in_fix"]==0][columns] - weights_median1, axis=0), axis=0)
    return weights_mad # / weights_mad1


def evaluate_fold(df, Y):
    df = df.copy()
    df.index.names = ["bid", "fid"]
    r = df[["used_in_fix", "f1"]].copy(deep=False)
    r["result"] = Y
    min_fix_result = r[r["used_in_fix"] == 1.0]["result"].min()
    minimal_reasonable_set = r[r["result"] >= min_fix_result].copy()
    acc, m_a_p, mrr, k_range = get_no_tie_on_df(minimal_reasonable_set)
    # eprint()
    # eprint("evaluate fold")
    # eprint(len(np.unique(Y)))
    # eprint(np.unique(Y))
    # eprint("acc", acc)
    # eprint("map", m_a_p)
    # eprint("mrr", mrr)
    # eprint()
    return m_a_p


def weights_on_df(method, df, columns):
    weights = method(df, columns)
    return method.__name__, weights


def eval_weights(m_name, weights, df, columns):
    Y = np.dot(df[columns], weights)
    return m_name, (weights, evaluate_fold(df, Y))


def fold_check(method, df, columns):
    weights = method(df, columns)
    Y = np.dot(df[columns], weights)
    return method.__name__, (weights, evaluate_fold(df, Y))


def fold_check_combination(w1, w2, df):
    weights = w1[1][0] + w2[1][0]
    Y = np.dot(df[feature_columns], weights)
    return w1[0] + w2[0], (weights, evaluate_fold(df, Y))


def size_selectf_only_fixes(df, score):
    used_in_fix = df["used_in_fix"] == 1
    ret = used_in_fix
    # eprint("11>>>>>>>>>",ret.sum())

    G = df[["used_in_fix", "f1"]].copy(deep=False)
    G["score"] = score

    # top20_per_bug = G["score"].groupby(level=0).apply(lambda x: x.nlargest(10).min())
    # top20 = top20_per_bug.nsmallest(1000)
    # eprint("12>>>>>>>>>",)
    # ret |= G['score'].isin(top20)

    # ret |= G['score'].isin(G['score'].nlargest(500))
    t = G[G['score'] > 0]['score']
    tm = t.nsmallest(int(0.25 * used_in_fix.sum())).max()
    ret |= G['score'] <= tm
    ret &= G['score'] > 0

    # eprint("12>>>>>>>>>",ret.sum(), int(0.30*used_in_fix.sum()), tm)
    return ret


def size_selectf_only_fixes_p(df, score, perc=0.25, smallest=True, largest=False):
    used_in_fix = df["used_in_fix"] == 1
    ret = used_in_fix

    G = df[["used_in_fix"]].copy(deep=False)
    G["score"] = score

    t = G[G['score'] > 0]['score']
    if smallest:
        tm = t.nsmallest(int(perc * used_in_fix.sum())).max()
        ret |= G['score'] <= tm
    if largest:
        tm = t.nlargest(int(perc * used_in_fix.sum())).min()
        ret |= G['score'] >= tm

    ret &= G['score'] > 0

    return ret


def get_skmodels():
    sgd_loss = [
        "squared_loss",
        "huber",
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ]
    sgd_penalty = ["none", "l2", "l1", "elasticnet"]
    alpha = 10.0**-np.arange(4, 5)
    return [
        SGDRegressor(max_iter=1000, shuffle=False, loss=l, penalty=p, alpha=a)
        for l, p, a in product(sgd_loss, sgd_penalty, alpha)
    ]


def normal_score(df, columns, weights):
    score = np.dot(df[columns], weights)
    return score


def cut_fit_predict(df, df_test, columns, score, score_fixed, cut_method, reg_model):
    cut_set = cut_method(df, score)
    X = df[cut_set]

    reg_model.fit(X[columns], score_fixed[cut_set])
    Y = reg_model.predict(df_test[columns])

    return evaluate_fold(df_test, Y)


def size_selectf_only_fixes_p_perc_05(df, score):
    return size_selectf_only_fixes_p(df, score, perc=0.05)


def size_selectf_only_fixes_p_perc_10(df, score):
    return size_selectf_only_fixes_p(df, score, perc=0.10)


def size_selectf_only_fixes_p_perc_15(df, score):
    return size_selectf_only_fixes_p(df, score, perc=0.15)


def size_selectf_only_fixes_p_perc_20(df, score):
    return size_selectf_only_fixes_p(df, score, perc=0.20)


def size_selectf_only_fixes_p_perc_25(df, score):
    return size_selectf_only_fixes_p(df, score, perc=0.25)


def size_selectf_only_fixes_p_perc_30(df, score):
    return size_selectf_only_fixes_p(df, score, perc=0.30)


class Adaptive_Process(object):
    def __init__(self):
        self.weights_methods = [
            weights_AdaBoostClassifier,
            weights_ExtraTreesClassifier,
            weights_GradientBoostingClassifier,
            weights_const,
            weights_variance,
            weights_chi2,
            weights_mutual_info_classif,
            weights_FastICA,
            weights_kruskal_classif,
            weights_ttest_ind_classif,
            weights_levene_median,
            weights_mean_var,
            weights_maximum_absolute_deviation,
        ]
        self.weights = {}

        self.reg_models = []
        self.reg_models.extend(get_skmodels())

        # Works for aspectj, birt, swt
        self.cut_methods = []
        # self.cut_methods.append(size_selectf_only_fixes)

        self.cut_methods.append(size_selectf_only_fixes_p_perc_05)
        self.cut_methods.append(size_selectf_only_fixes_p_perc_10)
        self.cut_methods.append(size_selectf_only_fixes_p_perc_15)
        self.cut_methods.append(size_selectf_only_fixes_p_perc_20)
        self.cut_methods.append(size_selectf_only_fixes_p_perc_25)
        self.cut_methods.append(size_selectf_only_fixes_p_perc_30)

        self.score_methods = []
        self.score_methods.append(normal_score)

        self.score_methods_map = {m.__name__: m for m in self.score_methods}
        self.cut_methods_map = {m.__name__: m for m in self.cut_methods}
        self.reg_models_map = {str(m): m for m in self.reg_models}

        self.name = "Adaptive"
        self.first_fold_processed = False

        self.enforce_relearning = True

        self.use_multiplied_features = False

        self.use_aggregated_features = False
        self.drop_not_aggregated_features = False

        self.use_prescoring_always = False
        self.use_reg_model_always = True

        self.use_prescoring_cross_validation = True
        self.use_training_cross_validation = True
        self.cross_validation_fold_number = 2

        self.previous_models = []

        self.pca = None
        self.reg_model = None
        self.cut_method = None
        self.score_method = None
        self.weights = None
        self.columns = None

        self.training_time_list = []

        self.prescoring_log = []
        self.best_prescoring_log = []
        self.regression_log = []
        self.best_regression_log = []

    def compute_weights(self, df, columns):
        if self.use_prescoring_cross_validation:
            kfold = KFold(n_splits=self.cross_validation_fold_number, random_state=None, shuffle=False)
            partial_result_dict = defaultdict(list)
            for train_index, test_index in kfold.split(df):
                kdf = df.iloc[train_index]
                weights = Parallel(n_jobs=-1)(
                    delayed(weights_on_df)(m, kdf, columns) for m in tqdm.tqdm(self.weights_methods)
                )
                kdf_test = df.iloc[test_index]
                weights_results = Parallel(n_jobs=-1)(
                    delayed(eval_weights)(m, w, kdf_test, columns) for m, w in tqdm.tqdm(weights)
                )
                weights_results_dict = dict(weights_results)
                for m_name in weights_results_dict:
                    partial_result_dict[m_name].append(weights_results_dict[m_name])
            results = {}
            for m_name in partial_result_dict:
                # print(m_name)
                # print(partial_result_dict[m_name])
                values = partial_result_dict[m_name]
                weights_avg = []
                eval_avg = []
                for value in values:
                    weights_avg.append(value[0])
                    eval_avg.append(value[1])
                weights_avg = np.mean(weights_avg, axis=0)
                eval_avg = np.mean(eval_avg)
                # print(weights_avg)
                # print(eval_avg)
                results[m_name] = (weights_avg, eval_avg)
                # exit(0)
            self.weights = results
        else:
            results = Parallel(n_jobs=-1)(
                delayed(fold_check)(m, df, columns) for m in tqdm.tqdm(self.weights_methods)
            )
            self.weights = dict(results)

    def add_multiplied_features(self, df, columns):
        similarity_features = ['f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14']
        graph_features = ['f15', 'f16', 'f17', 'f18', 'f19']
        for s_feature in similarity_features:
            for g_feature in graph_features:
                v = df[s_feature] * df[g_feature]
                n = s_feature+g_feature
                columns.append(n)
                df[n] = v
        return df, columns

    def add_combined_features(self, df, columns):
        feature_groups = [
            ['f7', 'f8', 'f9', 'f10'],
            ['f11', 'f12', 'f13', 'f14'],
            ['f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14'],
            ['f15', 'f16', 'f17', 'f18', 'f19'],
            ['f17', 'f18', 'f19'],
            ['f15', 'f16'],
        ]
        for feature_group in feature_groups:
            max_per_row = df[feature_group].max(axis=1)
            max_feature_name = str(feature_group)+'max'
            df[max_feature_name] = max_per_row
            columns.append(max_feature_name)

            median_per_row = df[feature_group].median(axis=1)
            median_feature_name = str(feature_group)+'median'
            df[median_feature_name] = median_per_row
            columns.append(median_feature_name)

            min_per_row = df[feature_group].min(axis=1)
            min_feature_name = str(feature_group)+'min'
            df[min_feature_name] = min_per_row
            columns.append(min_feature_name)

        if self.drop_not_aggregated_features:
            selected_feature_columns = feature_columns.copy()
            selected_feature_columns.remove("f1")
            selected_feature_columns.remove("f2")
            selected_feature_columns.remove("f3")
            selected_feature_columns.remove("f4")
            selected_feature_columns.remove("f5")
            selected_feature_columns.remove("f6")
            df = df.drop(columns=selected_feature_columns)
            for feature_column in selected_feature_columns:
                columns.remove(feature_column)

        return df, columns

    def adapt_process(self, df, columns):
        eprint("=============== Weights Select")
        self.compute_weights(df, columns)

        w_maks = 0
        w_method = None
        w_weights = None
        for k, v in self.weights.items():
            self.prescoring_log.append((k, v[1]))
            if v[1] > w_maks:
                w_maks = v[1]
                w_method = k
                w_weights = v[0]

        self.weights = w_weights
        self.weights_score = w_maks
        eprint(w_method, w_weights, w_maks)
        self.best_prescoring_log.append((w_method, w_maks))
        eprint("===============")

        eprint("=============== Size and regression model select")

        results = Parallel(n_jobs=-1)(
            delayed(self._train)(df, columns, w_weights, score_method, reg_model, cut_method)
            for score_method, reg_model, cut_method in tqdm.tqdm(
                product(self.score_methods, self.reg_models, self.cut_methods)
            )
        )

        res_max = 0
        for res in results:
            current_name = res[0]
            current_cut_function = res[1]
            current_score_function = res[2]
            current_score = res[3]
            current_reg_model = self.reg_models_map[current_name]

            name = self.prepare_regressor_name(current_reg_model)
            self.regression_log.append((name,  current_cut_function, current_score_function, current_score))
            if res[3] > res_max:
                res_max = res[3]
                self.reg_model_name = res[0]
                self.cut_method_name = res[1]
                self.score_method_name = res[2]

        self.reg_model = self.reg_models_map[self.reg_model_name]
        self.cut_method = self.cut_methods_map[self.cut_method_name]
        self.score_method = self.score_methods_map[self.score_method_name]

        self.reg_model_score = res_max
        current_reg_model = self.reg_model
        name = self.prepare_regressor_name(current_reg_model)
        self.best_regression_log.append((name, self.cut_method_name, self.score_method_name, self.reg_model_score))

        eprint(res_max, self.reg_model_name, self.cut_method_name, self.score_method_name)
        eprint("===============")

    def prepare_regressor_name(self, current_reg_model):
        if isinstance(current_reg_model, SGDRegressor):
            name = 'SGDRegressor' + '_' + current_reg_model.loss + '_' + current_reg_model.penalty + '_' + \
                   str(current_reg_model.alpha) + '_' + str(current_reg_model.shuffle)
        else:
            name = self.reg_model_name
        return name

    def transform_with_pca(self, df, columns):
        self.pca = TruncatedSVD(n_components=5)
        self.pca.fit(df[df['used_in_fix'] == 1][columns])
        ndf = pd.DataFrame(self.pca.transform(df[columns]))
        ndf.index = df.index
        feature_pca = ndf.columns
        ndf['used_in_fix'] = df['used_in_fix']
        df = ndf
        columns = feature_pca
        return df, columns

    def _train(self, df, columns, weights, score_method, reg_model, cut_method):

        score = score_method(df, columns, weights)
        score_fixed = score + df["used_in_fix"] * np.max(score)

        if self.use_training_cross_validation:
            # eprint("Attempting cross validation")
            # eprint("X type", type(X))
            # eprint("X shape", X[feature_columns].shape)
            # eprint("score_fixed[cut_set] type", type(score_fixed[cut_set]))
            # eprint("score_fixed[cut_set] shape", score_fixed[cut_set].shape)
            # eprint("cross validation fold number", self.cross_validation_fold_number)

            kfold = KFold(n_splits=self.cross_validation_fold_number, random_state=None, shuffle=False)
            partial_eval_results = []
            for train_index, test_index in kfold.split(df):
                kdf = df.iloc[train_index]
                kscore = score[train_index]
                kscore_fixed = score_fixed.iloc[train_index]

                kdf_test = df.iloc[test_index]
                pres = cut_fit_predict(kdf, kdf_test, columns, kscore, kscore_fixed, cut_method, reg_model)

                # cut_set = cut_method(kdf, kscore)
                # kX = kdf[cut_set]
                # reg_model.fit(kX[feature_columns], kscore_fixed[cut_set])
                # Y = reg_model.predict(df[feature_columns].iloc[test_index])
                # partial_eval_result = evaluate_fold(df.iloc[test_index], Y)

                partial_eval_results.append(pres)
            eval_result = np.mean(partial_eval_results)
            return str(reg_model), cut_method.__name__, score_method.__name__, eval_result
        else:
            return str(reg_model), cut_method.__name__, score_method.__name__, cut_fit_predict(df, df, columns, score, score_fixed, cut_method, reg_model)

    def train(self, df):
        before_training = default_timer()
        columns = feature_columns.copy()

        if self.use_multiplied_features:
            df, columns = self.add_multiplied_features(df, columns)

        if self.use_aggregated_features:
            df, columns = self.add_combined_features(df, columns)

        if not self.first_fold_processed or self.enforce_relearning:
            self.adapt_process(df, columns)
            self.first_fold_processed = True

        self._train(df, columns, self.weights, self.score_method, self.reg_model, self.cut_method)
        self.previous_models.append(self.weights)
        self.columns = columns

        after_training = default_timer()
        total_training = after_training - before_training
        self.training_time_list.append((total_training,
                                        df.index.get_level_values(0).unique().shape[0],
                                        df.index.get_level_values(1).unique().shape[0]))

        return self.reg_model

    def predict(self, clf, df):
        df.index.names = ["bid", "fid"]
        columns = self.columns.copy()
        df_columns = feature_columns.copy()

        if self.use_multiplied_features:
            df, df_columns = self.add_multiplied_features(df, df_columns)

        if self.use_aggregated_features:
            df, df_columns = self.add_combined_features(df, df_columns)

        X = df[columns].values

        # Check if weights method gives better results on training
        if not self.use_prescoring_always and (self.reg_model_score >= self.weights_score or self.use_reg_model_always):
            result = clf.predict(X)
        else:
            result = np.dot(X, self.weights)

        # result = self.post_score(df, X, self.W)
        r = df[["used_in_fix", "f1"]].copy(deep=False)
        r["result"] = result

        return r


if __name__ == "__main__":
    main()
