#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <data_prefix>
"""

import json
from timeit import default_timer

import datetime
import pickle
import sys
from multiprocessing import Pool
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

    convert_tf_idf_for_each_fix(fixes_list, data_prefix)

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
        commit_dates.append((sha, commit_date))

    sorted_commit_dates = sorted(commit_dates, key=itemgetter(1))
    sorted_commits = [commit_date[0] for commit_date in sorted_commit_dates]
    return sorted_commits


def extract_fixes_list(bug_report_file_path):
    bug_reports = load_bug_reports(bug_report_file_path)
    return sort_bug_reports_by_commit_date(bug_reports)


def convert_tf_idf_for_each_fix(fixes_list, data_prefix):
    work = []
    for fixing_commit in fixes_list:
        work.append((data_prefix, fixing_commit))
    
    pool = Pool(12, maxtasksperchild=1)
    r = list(tqdm(pool.imap(_f, work), total=len(work)))
    print("r", len(r))
#debug    _f(work[0])


def _f(args):
    return convert_tf_idf(args[0], args[1])
 

def convert_tf_idf(data_prefix, bug_report_full_sha):
    bug_report_files_collection_db = UnQLite(data_prefix+"_bug_report_files_collection_db", flags = 0x00000100 | 0x00000001)
    current_files = pickle.loads(bug_report_files_collection_db[bug_report_full_sha])
    bug_report_files_collection_db.close() 

    bug_report_id = bug_report_full_sha[0:7]

    shas = current_files['shas']
    class_name_lookup = current_files['class_name_to_sha']

    ast_index_collection = UnQLite(data_prefix+"_ast_index_collection_index_db", flags = 0x00000100 | 0x00000001)
    data = sparse.load_npz(data_prefix+'_raw_count_data.npz')

    data_to_tf_idf = []
    lookups = {}
    n_rows = 0
    for sha in shas:
        current_indexes = pickle.loads(ast_index_collection[sha])
#        print(sha)
#        print(current_indexes)
        (matrix, lookup) = extract_ast(data, current_indexes)
#        print(lookup)
#        print(matrix.shape)
        current_index = n_rows
        data_to_tf_idf.append(matrix)
        for k in lookup:
            lookup[k] += current_index
        lookups[sha] = lookup
        n_rows += matrix.shape[0]

    ast_index_collection.close()

    bug_report_index_collection = UnQLite(data_prefix+"_bug_report_index_collection_index_db", flags = 0x00000100 | 0x00000001)
    current_bug_report_indexes = pickle.loads(bug_report_index_collection[bug_report_id])
    bug_report_index_collection.close()

    bug_report_matrix, bug_report_lookup = extract_bug_report(data, current_bug_report_indexes)
    current_index = n_rows
    data_to_tf_idf.append(bug_report_matrix)
    for k in bug_report_lookup:
        bug_report_lookup[k] += current_index
    lookups[bug_report_id] = bug_report_lookup
    n_rows += bug_report_matrix.shape[0]
    
    data_matrix = sparse.vstack(data_to_tf_idf)

    transformer = TfidfTransformer()
    tf_idf_data = transformer.fit_transform(data_matrix)
#    print("tf_idf_data shape",tf_idf_data.shape)
    sparse.save_npz(data_prefix+'_'+bug_report_id+'_tf_idf_data', tf_idf_data)
    with open(data_prefix+'_'+bug_report_id+'_tf_idf_index_lookup', 'w') as outfile:
        json.dump(lookups, outfile)


def extract_ast(data, current_indexes):
    lookup = {}

    i = 0
    source_row = data[current_indexes['source'], :]
    lookup['source'] = i
    i += 1

    methods_rows = data[current_indexes['methodsStart']:current_indexes['methodsEnd']+1,:]
    lookup['methodsStart'] = i
    i += methods_rows.shape[0] - 1
    lookup['methodsEnd'] = i  
    i += 1

    class_names_rows = data[current_indexes['classNamesStart']:current_indexes['classNamesEnd']+1,:]
    lookup['classNamesStart'] = i
    i += class_names_rows.shape[0] - 1
    lookup['classNamesEnd'] = i  
    i += 1

    method_names_rows = data[current_indexes['methodNamesStart']:current_indexes['methodNamesEnd']+1,:]
    lookup['methodNamesStart'] = i
    i += method_names_rows.shape[0] - 1
    lookup['methodNamesEnd'] = i
    i += 1

    variable_names_rows = data[current_indexes['variableNamesStart']:current_indexes['variableNamesEnd']+1,:]
    lookup['variableNamesStart'] = i
    i += variable_names_rows.shape[0] - 1
    lookup['variableNamesEnd'] = i 
    i += 1

    comments_rows = data[current_indexes['commentsStart']:current_indexes['commentsEnd']+1,:]
    lookup['commentsStart'] = i
    i += comments_rows.shape[0] - 1
    lookup['commentsEnd'] = i 
    i += 1

    rows = [source_row, methods_rows, class_names_rows, method_names_rows, variable_names_rows, comments_rows]
    matrix = sparse.vstack(rows)

    return (matrix, lookup)


def extract_bug_report(data, bug_report_indexes):
    lookup = {}

    i = 0
    summary_row = data[bug_report_indexes['summary'], :]
    lookup['summary'] = i
    i += 1

    description_row = data[bug_report_indexes['description'], :]
    lookup['description'] = i
    i += 1

    report_row = data[bug_report_indexes['report'], :]
    lookup['report'] = i
    i += 1

    rows = [summary_row, description_row, report_row]
    matrix = sparse.vstack(rows)

    return (matrix, lookup)

if __name__ == '__main__':
    main()
