#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <data_prefix>
"""

import json
from timeit import default_timer

import datetime
import numpy as np
import pickle
import sys
from multiprocessing import Pool, Manager
from operator import itemgetter
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
from unqlite import UnQLite

from date_utils import convert_commit_date


def main():
    print("Start", datetime.datetime.now().isoformat()) 
    before = default_timer()
    bug_report_file_path = sys.argv[1]
    print("bug report file path", bug_report_file_path)
    data_prefix = sys.argv[2]
    print("data prefix", data_prefix)

    fixes_list = extract_fixes_list(bug_report_file_path)

    calculate_feature_3_for_each_fix(fixes_list, data_prefix, bug_report_file_path)

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
        sha = bug_reports[commit]['commit']['metadata']['sha'].replace('commit ','').strip()
        commit_date = convert_commit_date(bug_reports[commit]['commit']['metadata']['date'].replace('Date:','').strip())
        commit_dates.append((commit, sha, commit_date))

    sorted_commit_dates = sorted(commit_dates, key=itemgetter(1))
    sorted_commits = [(entry[0], entry[1]) for entry in sorted_commit_dates]
    return sorted_commits


def extract_fixes_list(bug_report_file_path):
    bug_reports = load_bug_reports(bug_report_file_path)
    return sort_bug_reports_by_commit_date(bug_reports)


def calculate_feature_3_for_each_fix(fixes_list, data_prefix, bug_report_file_path):

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


def _f(args):
    return calculate_feature_3(args[0], args[1], args[2], args[3])
 

def calculate_feature_3(data_prefix, bug_report_id, bug_report_full_sha, bug_reports):
    bug_report_files_collection_db = UnQLite(data_prefix+"_bug_report_files_collection_db", flags = 0x00000100 | 0x00000001)
    current_files = pickle.loads(bug_report_files_collection_db[bug_report_full_sha])
    bug_report_files_collection_db.close() 

    shas = current_files['shas']
    sha_to_file_name = current_files['sha_to_file_name']

    data = sparse.load_npz(data_prefix+'_raw_count_data.npz')
    row_length = data.shape[1]

    current_bug_report = bug_reports[bug_report_id]

    bug_report_index_collection = UnQLite(data_prefix+"_bug_report_index_collection_index_db", flags = 0x00000100 | 0x00000001)
    current_bug_report_summary_index = pickle.loads(bug_report_index_collection[bug_report_id[0:7]])['summary']

    feature_3_data_list = [] 
    feature_3_lookup = {}

    if 'views' in current_bug_report and 'bug_fixing' in current_bug_report['views']:
        bug_fixing_view = current_bug_report['views']['bug_fixing']
        current_index = 0
        for sha in shas:
            current_file_name = sha_to_file_name[sha]
            if current_file_name in bug_fixing_view:
                related_bug_reports = bug_fixing_view[current_file_name]['br']
                # print("Present",sha)
                # print(related_bug_reports)
                bug_report_history = combine(related_bug_reports, data, bug_report_index_collection)
            else:
                bug_report_history = np.zeros((1, row_length))
                # print("Not present",sha)
            feature_3_data_list.append(bug_report_history)
            feature_3_lookup[sha] = current_index
            current_index += 1
    else:
        current_index = 0
        for sha in shas:
            bug_report_history = np.zeros((1, row_length))
            feature_3_data_list.append(bug_report_history)
            feature_3_lookup[sha] = current_index
            current_index += 1

    bug_report_index_collection.close()

    feature_3_data_list.append(data[current_bug_report_summary_index, :])
  
    feature_3_data = sparse.vstack(feature_3_data_list)

    transformer = TfidfTransformer()
    feature_3_tf_idf_data = transformer.fit_transform(feature_3_data)

    sparse.save_npz(data_prefix+'_'+bug_report_id[0:7]+'_feature_3_data', feature_3_tf_idf_data)
    with open(data_prefix+'_'+bug_report_id[0:7]+'_feature_3_index_lookup', 'w') as outfile:
        json.dump(feature_3_lookup, outfile)

    return (bug_report_id)


def combine(related_bug_reports, data, bug_report_index_collection):
    indexes = list(map(lambda bug_report_id: get_bug_report_summary_index(bug_report_id, bug_report_index_collection), related_bug_reports))
    # print(indexes)
    return np.sum(data[indexes, :], axis = 0)


def get_bug_report_summary_index(bug_report_id, bug_report_index_collection):
    # print("Bug_report_id",bug_report_id)
    # if bug_report_id[0:7] in bug_report_index_collection:
    #     print("In bug_report_index_collection")
    return pickle.loads(bug_report_index_collection[bug_report_id[0:7]])['summary']

if __name__ == '__main__':
    main()
