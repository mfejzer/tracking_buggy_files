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
from sklearn.metrics.pairwise import cosine_similarity
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

    bug_reports = load_bug_reports(bug_report_file_path)

    process(bug_reports, data_prefix, bug_report_file_path)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time", total)


def load_bug_reports(bug_report_file_path):
    """load bug report file (the one generated from xml)"""
    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        return bug_reports


def load_vectorized_data(data_prefix):
    file_path = data_prefix + '_all_data.npz'
    print("vectorized data file path", file_path)
    vectorized_data = sparse.load_npz(file_path).tocsr()
    print("vectorized data shape", vectorized_data.shape)
    return vectorized_data


def sort_bug_reports_by_commit_date(bug_reports):
    commit_dates = []
    for index, commit in enumerate(tqdm(bug_reports)):
        sha = bug_reports[commit]['commit']['metadata']['sha'].replace('commit ', '').strip()
        commit_date = convert_commit_date(
            bug_reports[commit]['commit']['metadata']['date'].replace('Date:', '').strip())
        commit_dates.append((sha, commit_date))

    sorted_commit_dates = sorted(commit_dates, key=itemgetter(1))
    sorted_commits = [commit_date[0] for commit_date in sorted_commit_dates]
    return sorted_commits


def load_bug_report(vectorized_data, bug_report_indexes, bug_report_id):
    # index_dict = pickle.loads(bug_report_indexes[bug_report_id[0:7]])
    index_dict = bug_report_indexes[bug_report_id]
    report_index = index_dict['report']
    vectorized_report = vectorized_data[report_index, :]

    summary_index = index_dict['summary']
    vectorized_summary = vectorized_data[summary_index, :]

    description_index = index_dict['description']
    vectorized_description = vectorized_data[description_index, :]

    return vectorized_report, vectorized_summary, vectorized_description


def feature_1(report, data, source_index, method_start_index, method_end_index):
    sources = data[source_index:method_end_index + 1, :]
    similarities = cosine_similarity(report, sources)

    return np.amax(similarities)


def feature_2(report, data, enriched_api_indexes, current_file_sha):
    file_enriched_api = enriched_api_indexes[current_file_sha]
    enriched_api_start = file_enriched_api['enrichedApiStart']
    enriched_api_end = file_enriched_api['enrichedApiEnd']
    return feature_sim(report, data, enriched_api_start, enriched_api_end)


def feature_sim(document, data, start_index, end_index):
    if start_index == end_index + 1:
        return 0.0
    sources = data[start_index:end_index + 1, :]
    similarities = cosine_similarity(document, sources)

    return np.amax(similarities)


def feature_3(data, lookup, sha):
    if sha not in lookup:
        return 0.0
    file_index = lookup[sha]
    report_summary_index = data.shape[0] - 1
    return cosine_similarity(data[report_summary_index, :], data[file_index, :])[0][0]


def feature_4(current_bug_report_summary, ast_cache_collection, sha):
    class_names = pickle.loads(ast_cache_collection[sha])['classNames']
    class_names = remove_package_names(class_names)

    lengths = [0]
    for class_name in class_names:
        if class_name in current_bug_report_summary:
            lengths.append(len(class_name))
    max_length = max(lengths)
    return max_length


def remove_package_names(class_names):
    return map(lambda x: remove_package_name(x), class_names)


def remove_package_name(class_name):
    return class_name.split(".")[-1]


def process(bug_reports, data_prefix, bug_report_file_path):
    sorted_commits = sort_bug_reports_by_commit_date(bug_reports)

    with open(data_prefix + '_features_5_6_max', 'r') as maxfile:
        features_5_6_max = json.load(maxfile)
    max_frequency = features_5_6_max['max_frequency']

    work = []
    for fixing_commit in sorted_commits:
        work.append((data_prefix, fixing_commit, bug_report_file_path, max_frequency))

    pool = Pool(12, maxtasksperchild=1)
    r = list(tqdm(pool.imap(_f, work), total=len(work)))
    print("r", len(r))
#    _f(work[0])


def _f(args):
    return process_bug_report(args[0], args[1], args[2], args[3])


def retrieve_summary(bug_reports, bug_report_full_sha):
    i = 7
    partial_key = bug_report_full_sha[0:i]
    while partial_key not in bug_reports or i == len(bug_report_full_sha):
        i += 1
        partial_key = bug_report_full_sha[0:i]
    return bug_reports[partial_key]


def process_bug_report(data_prefix, bug_report_full_sha, bug_report_file_path, max_frequency):
    bug_report_files_collection_db = UnQLite(data_prefix + "_bug_report_files_collection_db",
                                             flags=0x00000100 | 0x00000001)
    current_files = pickle.loads(bug_report_files_collection_db[bug_report_full_sha])
    bug_report_files_collection_db.close()

    shas = current_files['shas']
    sha_to_file_name = current_files['sha_to_file_name']

    bug_report_id = bug_report_full_sha[0:7]
    vectorized_data = sparse.load_npz(data_prefix + '_' + bug_report_id + '_tf_idf_data.npz')
    with open(data_prefix + '_' + bug_report_id + '_tf_idf_index_lookup', 'r') as index_lookup_file:
        lookups = json.load(index_lookup_file)

    enriched_api_data, enriched_api_indexes = load_enriched_api(data_prefix, bug_report_id)
    enriched_report = enriched_api_data[-1, :]

    (vectorized_report, vectorized_summary, vectorized_description) = load_bug_report(vectorized_data, lookups,
                                                                                      bug_report_id)

    ast_cache_collection = UnQLite(data_prefix + "_ast_cache_collection_db", flags=0x00000100 | 0x00000001)
    bug_reports = load_bug_reports(bug_report_file_path)
    if bug_report_id in bug_reports:
        current_bug_report_summary = bug_reports[bug_report_id]['bug_report']['summary']
    else:
        current_bug_report_summary = retrieve_summary(bug_reports, bug_report_full_sha)['bug_report']['summary']

    feature_3_data = sparse.load_npz(data_prefix + '_' + bug_report_id + '_feature_3_data.npz')
    with open(data_prefix + '_' + bug_report_id + '_feature_3_index_lookup', 'r') as feature_3_file:
        feature_3_file_lookup = json.load(feature_3_file)

    graph_data = sparse.load_npz(data_prefix + '_' + bug_report_id + '_graph_features_data.npz').tocsr()
    with open(data_prefix + '_' + bug_report_id + '_graph_features_index_lookup', 'r') as graph_lookup_file:
        graph_lookup = json.load(graph_lookup_file)

    features_5_6_data = sparse.load_npz(data_prefix + '_' + bug_report_id[0:7] + '_features_5_6_data.npz').tocsr()
    with open(data_prefix + '_' + bug_report_id[0:7] + '_features_5_6_index_lookup', 'r') as feaures_5_6_lookup_file:
        features_5_6_lookup = json.load(feaures_5_6_lookup_file)

    if bug_report_id in bug_reports:
        fixed_filenames = bug_reports[bug_report_id[0:7]]['commit']['diff'].keys()
    else:
        fixed_filenames = retrieve_summary(bug_reports, bug_report_full_sha)['commit']['diff'].keys()

    features = []
    features_files = []

    for file_index, current_file_sha in enumerate(shas):
        current_lookup = lookups[current_file_sha]
        source_index = current_lookup['source']

        method_source_start_index = current_lookup['methodsStart']
        method_source_end_index = current_lookup['methodsEnd']

        class_start_index = current_lookup['classNamesStart']
        class_end_index = current_lookup['classNamesEnd']

        method_names_start_index = current_lookup['methodNamesStart']
        method_names_end_index = current_lookup['methodNamesEnd']

        variable_start_index = current_lookup['variableNamesStart']
        variable_end_index = current_lookup['variableNamesEnd']

        comment_start_index = current_lookup['commentsStart']
        comment_end_index = current_lookup['commentsEnd']

        current_graph_lookup = graph_lookup[current_file_sha]

        current_features_5_6 = features_5_6_lookup[current_file_sha]

        current_file_name = sha_to_file_name[current_file_sha]

        f1 = feature_1(vectorized_report, vectorized_data, source_index, method_source_start_index,
                       method_source_end_index)
        f2 = feature_2(enriched_report, enriched_api_data, enriched_api_indexes, current_file_sha)
        f3 = feature_3(feature_3_data, feature_3_file_lookup, current_file_sha)
        f4 = feature_4(current_bug_report_summary, ast_cache_collection, current_file_sha)

        f5 = (features_5_6_data[current_features_5_6, 0])
        f6 = (features_5_6_data[current_features_5_6, 1]) / max_frequency

        f7 = feature_sim(vectorized_summary, vectorized_data, class_start_index, class_end_index)
        f8 = feature_sim(vectorized_summary, vectorized_data, method_names_start_index, method_names_end_index)
        f9 = feature_sim(vectorized_summary, vectorized_data, variable_start_index, variable_end_index)
        f10 = feature_sim(vectorized_summary, vectorized_data, comment_start_index, comment_end_index)

        f11 = feature_sim(vectorized_description, vectorized_data, class_start_index, class_end_index)
        f12 = feature_sim(vectorized_description, vectorized_data, method_names_start_index, method_names_end_index)
        f13 = feature_sim(vectorized_description, vectorized_data, variable_start_index, variable_end_index)
        f14 = feature_sim(vectorized_description, vectorized_data, comment_start_index, comment_end_index)

        f15 = graph_data[current_graph_lookup, 0]
        f16 = graph_data[current_graph_lookup, 1]
        f17 = graph_data[current_graph_lookup, 2]
        f18 = graph_data[current_graph_lookup, 3]
        f19 = graph_data[current_graph_lookup, 4]

        if current_file_name in fixed_filenames:
            used_in_fix = 1.0
        else:
            used_in_fix = 0.0

        features.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, used_in_fix])
        features_files.append(current_file_sha)

    ast_cache_collection.close()

    sparse_features = sparse.csr_matrix(features)
    sparse.save_npz(data_prefix+'_'+bug_report_id+'_features', sparse_features)
    with open(data_prefix+'_'+bug_report_id+'_files', 'w') as outfile:
        json.dump(features_files, outfile)


def load_enriched_api(data_prefix, commit):
    bug_report_id = commit[0:7]
    data = sparse.load_npz(data_prefix+'_'+bug_report_id+'_tfidf_enriched_api.npz')
    with open(data_prefix+'_'+bug_report_id+'_partial_enriched_api_index_lookup', 'r') as lookup_file:
        indexes = json.load(lookup_file)
    return (data, indexes)


if __name__ == '__main__':
    main()
