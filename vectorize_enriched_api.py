#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_report_file> <data_prefix>
"""

import json
from timeit import default_timer

import datetime
import numpy as np
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
    vectorize_enriched_api(fixes_list, data_prefix)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time",  total)


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


def find_supertype_shas(types, class_name_lookup, variable_sha):
    if variable_sha not in types:
        return []
#    variable_type = types[variable_sha]

    variable_type = pickle.loads(types[variable_sha])

    shas = []
    for name in variable_type['superclassNames']:
        if name in class_name_lookup:
           shas.append(class_name_lookup[name])
    for name in variable_type['interfaceNames']:
        if name in class_name_lookup:
           shas.append(class_name_lookup[name])
    return shas

def find_types_shas(types, class_name_lookup, sha):
    result = []
    to_check = [sha]
    while to_check:
        current_sha = to_check.pop(0)
        if current_sha not in result:
            result.append(current_sha)
            supertypes = find_supertype_shas(types, class_name_lookup, current_sha)
            to_check.extend(supertypes)
    return result

def get_indexes(asts, shas):
    indexes = []
    for sha in shas:
#        indexes.append(asts[sha]['source'])
        source_index = pickle.loads(asts[sha])['source']
        indexes.append(source_index)
    return indexes

def add_types_source_to_bug_report_data(data, data_prefix, class_name_lookup, ast_sha):
    asts = UnQLite(data_prefix+"_ast_index_collection_index_db", flags = 0x00000100 | 0x00000001)
    types = UnQLite(data_prefix+"_ast_types_collection_index_db", flags = 0x00000100 | 0x00000001)

#    current_type = types[ast_sha]
#    print "searching", ast_sha
    current_type = pickle.loads(types[ast_sha])
#    print "found", ast_sha
#    print current_type['methodVariableTypes']
#    exit(0)
    types_per_method = current_type['methodVariableTypes']

    cl = data.shape[1]

    current_index = 0 

    start = current_index
    enriched_apis = []
    for method_types in types_per_method:
        method_type_shas = []

        for method_type in method_types:
            if method_type in class_name_lookup:
                method_type_shas.append(class_name_lookup[method_type])

        supertypes_shas_per_type = [set(find_types_shas(types, class_name_lookup, s)) for s in method_type_shas]

        indexes = []
        for supertypes in supertypes_shas_per_type:
            indexes.extend(get_indexes(asts, supertypes))

        if indexes == []:
            method_enriched_api = sparse.coo_matrix(np.zeros(cl).reshape(1,cl))
        else:
            method_enriched_api = sparse.coo_matrix(np.sum((data[indexes,:]), axis = 0))

        enriched_apis.append(method_enriched_api)

    if enriched_apis == []:
        class_enriched_api = sparse.coo_matrix(np.zeros(cl).reshape(1,cl))
    else:
        class_enriched_api = sparse.coo_matrix(np.sum(enriched_apis, axis = 0))

    enriched_apis.append(class_enriched_api)

    current_index += len(enriched_apis) 

    asts.close()
    types.close()

    lookup = {}
    lookup['enrichedApiStart'] = start
    lookup['enrichedApiEnd'] = current_index - 1

    enriched_apis_matrix = sparse.vstack(enriched_apis)
    
    return (enriched_apis_matrix, lookup, ast_sha)

def vectorize_enriched_api(bug_report_fixing_commits, data_prefix):
    
    work = []
    for fixing_commit in bug_report_fixing_commits:
        work.append((data_prefix, fixing_commit))
    
    pool = Pool(12, maxtasksperchild=1)
    r = list(tqdm(pool.imap(_f, work), total=len(work)))
    print("r", len(r))

def _f(args):
    return extract_enriched_api(args[0], args[1])

def extract_enriched_api(data_prefix, bug_report_full_sha):
    data = sparse.load_npz(data_prefix+'_raw_count_data.npz')
    bug_report_files_collection_db = UnQLite(data_prefix+"_bug_report_files_collection_db", flags = 0x00000100 | 0x00000001)

    current_files = pickle.loads(bug_report_files_collection_db[bug_report_full_sha])
    bug_report_files_collection_db.close() 

    bug_report_id = bug_report_full_sha[0:7]

    shas = current_files['shas']
    class_name_lookup = current_files['class_name_to_sha']

    bug_report_data = []
    bug_report_lookup = {}

    n_rows = 0

    for ast_sha in shas:
        ast_data, lookup, current_ast_sha = add_types_source_to_bug_report_data(data, data_prefix, class_name_lookup, ast_sha)
        current_index = n_rows
        bug_report_data.append(ast_data)
        for k in lookup:
            lookup[k] += current_index
        bug_report_lookup[current_ast_sha] = lookup
        n_rows += ast_data.shape[0]

    bug_report_row = get_bug_report(data_prefix, data, bug_report_id)
    bug_report_data.append(bug_report_row)

    bug_report_data_matrix = sparse.vstack(bug_report_data)

    sparse.save_npz(data_prefix+'_'+bug_report_id+'_partial_enriched_api', bug_report_data_matrix)
    with open(data_prefix+'_'+bug_report_id+'_partial_enriched_api_index_lookup', 'w') as outfile:
        json.dump(bug_report_lookup, outfile)

    transformer = TfidfTransformer()
    tf_idf_data = transformer.fit_transform(bug_report_data_matrix)
    sparse.save_npz(data_prefix+'_'+bug_report_id+'_tfidf_enriched_api', tf_idf_data)

#    print "bug_report_id", bug_report_id

    return bug_report_id

def get_bug_report(data_prefix, vectorized_data, bug_report_id):
    bug_report_index_collection = UnQLite(data_prefix+"_bug_report_index_collection_index_db")
    bug_report = pickle.loads(bug_report_index_collection[bug_report_id])
    bug_report_index_collection.close()
    index = bug_report['report']
    return vectorized_data[index, :]


if __name__ == '__main__':
    main()
