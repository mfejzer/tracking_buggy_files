#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <data_prefix>
"""

import json
from timeit import default_timer

import datetime
import pickle
import sys
from multiprocessing import Pool, Manager
from operator import itemgetter
from scipy import sparse
from tqdm import tqdm
from unqlite import UnQLite

import numpy as np

from date_utils import convert_commit_date


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()
    bug_report_file_path = sys.argv[1]
    print("bug report file path", bug_report_file_path)
    data_prefix = sys.argv[2]
    print("data prefix", data_prefix)

    fixes_list = extract_fixes_list(bug_report_file_path)

    retrieve_features_5_6_for_each_fix(fixes_list, data_prefix, bug_report_file_path)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time ", total)


def load_bug_reports(bug_report_file_path):
    """load bug report file (the one generated from xml)"""
    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        return bug_reports


def sort_bug_reports_by_commit_date(bug_reports):
    commit_dates = []
    for index, commit in enumerate(tqdm(bug_reports)):
        sha = bug_reports[commit]['commit']['metadata']['sha'].replace('commit ', '').strip()
        commit_date = convert_commit_date(
            bug_reports[commit]['commit']['metadata']['date'].replace('Date:', '').strip())
        commit_dates.append((commit, sha, commit_date))

    sorted_commit_dates = sorted(commit_dates, key=itemgetter(1))
    sorted_commits = [(entry[0], entry[1]) for entry in sorted_commit_dates]
    return sorted_commits


def extract_fixes_list(bug_report_file_path):
    bug_reports = load_bug_reports(bug_report_file_path)
    return sort_bug_reports_by_commit_date(bug_reports)


def retrieve_features_5_6_for_each_fix(fixes_list, data_prefix, bug_report_file_path):
    manager = Manager()
    d = manager.dict()
    bug_reports = load_bug_reports(bug_report_file_path)
    d.update(bug_reports)

    work = []
    for fixing_commit in fixes_list:
        work.append((data_prefix, fixing_commit[0], fixing_commit[1], d))

    pool = Pool(12, maxtasksperchild=1)
    r = list(tqdm(pool.imap(_f, work), total=len(work)))
    # print(_f(work[0]))
    max_recency_array = np.array(list(map(lambda x: x[1], r)))
    max_frequency_array = np.array(list(map(lambda x: x[2], r)))

    max_recency = max_recency_array.max()
    max_frequency = max_frequency_array.max()

    print('max recency', max_recency)
    print('max frequency', max_frequency)

    features_5_6_max = {'max_recency': max_recency, 'max_frequency': max_frequency}
    with open(data_prefix + '_features_5_6_max', 'w') as outfile:
        json.dump(features_5_6_max, outfile)


def _f(args):
    return retrieve_features_5_6(args[0], args[1], args[2], args[3])


def retrieve_features_5_6(data_prefix, bug_report_id, bug_report_full_sha, bug_reports):
    bug_report_files_collection_db = UnQLite(data_prefix + "_bug_report_files_collection_db",
                                             flags=0x00000100 | 0x00000001)
    current_files = pickle.loads(bug_report_files_collection_db[bug_report_full_sha])
    bug_report_files_collection_db.close()

    shas = current_files['shas']
    sha_to_file_name = current_files['sha_to_file_name']

    current_bug_report = bug_reports[bug_report_id]

    features_5_6_data_list = []
    features_5_6_lookup = {}

    max_recency = 0.0
    max_frequency = 0.0

    if 'views' in current_bug_report and 'bug_fixing' in current_bug_report['views']:
        bug_fixing_view = current_bug_report['views']['bug_fixing']
        current_index = 0
        for sha in shas:
            current_file_name = sha_to_file_name[sha]
            if current_file_name in bug_fixing_view:
                recency = bug_fixing_view[current_file_name]['recency[30-day months]']
                frequency = bug_fixing_view[current_file_name]['frequency']
                features_5_6_data_list.append(sparse.coo_matrix([recency, frequency], shape=(1, 2)))
                features_5_6_lookup[sha] = current_index
                current_index += 1
                if recency > max_recency:
                    max_recency = recency
                if frequency > max_frequency:
                    max_frequency = frequency
            else:
                recency = 0.0
                frequency = 0.0
                features_5_6_data_list.append(sparse.coo_matrix([recency, frequency], shape=(1, 2)))
                features_5_6_lookup[sha] = current_index
                current_index += 1
    else:
        current_index = 0
        for sha in shas:
            recency = 0.0
            frequency = 0.0
            features_5_6_data_list.append(sparse.coo_matrix([recency, frequency], shape=(1, 2)))
            features_5_6_lookup[sha] = current_index
            current_index += 1

    features_5_6_data = sparse.vstack(features_5_6_data_list)

    sparse.save_npz(data_prefix + '_' + bug_report_id[0:7] + '_features_5_6_data', features_5_6_data)
    with open(data_prefix + '_' + bug_report_id[0:7] + '_features_5_6_index_lookup', 'w') as outfile:
        json.dump(features_5_6_lookup, outfile)

    return bug_report_id, max_recency, max_frequency


if __name__ == '__main__':
    main()
