#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_report_file> <data_prefix>

"""

import json
from timeit import default_timer

import datetime
import numpy as np
import sys
from multiprocessing import Pool
from operator import itemgetter
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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


def sort_bug_reports_by_id(bug_reports):
    dates = []
    for index, id in enumerate(tqdm(bug_reports)):
        date = bug_reports[id]['bug_report']['timestamp']
        dates.append((id, date))

    sorted_dates = sorted(dates, key=itemgetter(1))
    sorted_ids = [sorted_date[0] for sorted_date in sorted_dates]
    return sorted_ids


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
    sources = data[source_index:method_end_index+1, :]
    similarities = cosine_similarity(report, sources)

    return np.amax(similarities)


def feature_2(report, data, enriched_api_indexes, current_file_index):
    file_enriched_api = enriched_api_indexes[current_file_index]
    enriched_api_start = file_enriched_api['enrichedApiStart']
    enriched_api_end = file_enriched_api['enrichedApiEnd']
    return feature_sim(report, data, enriched_api_start, enriched_api_end)


def feature_sim(document, data, start_index, end_index):
    if start_index == end_index + 1:
        return 0.0
    sources = data[start_index:end_index+1, :]
    similarities = cosine_similarity(document, sources)

    return np.amax(similarities)


def feature_3(feature_3_data, feature_3_report_lookup, file_index):
    report_index = feature_3_report_lookup['report']
    packages_with_directory = file_index.replace('/', '.')
    for file in feature_3_report_lookup['files']:
        if packages_with_directory.endswith(file):
            feature_3_file_index = feature_3_report_lookup['files'][file]
            report_row = feature_3_data[report_index, :]
            file_row = feature_3_data[feature_3_file_index, :]
            return float(cosine_similarity(report_row, file_row))
    return 0.0


def feature_4(bug_report_summary, file_index):
    class_name = file_index.split('/')[-1].split('.')[0]
    if class_name in bug_report_summary:
        return len(class_name)
    else:
        return 0.0


def process(bug_reports, data_prefix, bug_report_file_path):
    sorted_ids = sort_bug_reports_by_id(bug_reports)

    work = []
    for bug_report_id in sorted_ids:
        work.append((data_prefix, bug_report_id, bug_report_file_path))
    # _f(work[0])
    # exit(0)

    pool = Pool(12, maxtasksperchild=1)
    r = list(tqdm(pool.imap(_f, work), total=len(work)))
    print("r", len(r))


def _f(args):
    return process_bug_report(args[0], args[1], args[2])


def process_bug_report(data_prefix, bug_report_id, bug_report_file_path):
    bug_reports = load_bug_reports(bug_report_file_path)
    bug_report = bug_reports[bug_report_id]

    vectorized_data = sparse.load_npz(data_prefix+'_raw_count_data.npz')
    with open(data_prefix+'_feature_names_dict', 'r') as infile:
        feature_names_lenghts_dict = json.load(infile)
    with open(data_prefix+'_file_index_lookup', 'r') as infile:
        file_index_lookup = json.load(infile)
    with open(data_prefix+'_bug_report_index_lookup', 'r') as infile:
        bug_report_index_lookup = json.load(infile)

    enriched_api_data = sparse.load_npz(data_prefix+'_tfidf_enriched_api.npz').tocsr()
    with open(data_prefix + '_partial_enriched_api_index_lookup') as infile:
        enriched_api_lookup = json.load(infile)
    with open(data_prefix + '_partial_enriched_api_bug_report_index_lookup') as infile:
        enriched_api_bug_reports_lookup = json.load(infile)

    graph_data = sparse.load_npz(data_prefix+'_graph_features_data.npz').tocsr()
    with open(data_prefix + '_graph_features_index_lookup', 'r') as infile:
        graph_lookup = json.load(infile)

    feature_3_data = sparse.load_npz(data_prefix + '_feature_3_data.npz')
    with open(data_prefix + '_feature_3_report_lookup', 'r') as infile:
        feature_3_report_lookup = json.load(infile)

    with open(data_prefix + '_feature_5_report_lookup', 'r') as infile:
        recency_lookup = json.load(infile)
    with open(data_prefix + '_feature_6_report_lookup', 'r') as infile:
        frequency_lookup = json.load(infile)

    (vectorized_report, vectorized_summary, vectorized_description) = load_bug_report(vectorized_data, bug_report_index_lookup, bug_report_id)

    enriched_report = enriched_api_data[enriched_api_bug_reports_lookup[bug_report_id], :]

    current_feature_3_report_lookup = feature_3_report_lookup[bug_report_id]

    current_bug_report_summary = bug_reports[bug_report_id]['bug_report']['summary']

    features = []
    features_files = []

    for file_index in file_index_lookup:
        current_lookup = file_index_lookup[file_index]
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

        current_graph_lookup = graph_lookup[file_index]

        current_recency_lookup = recency_lookup[bug_report_id]
        current_frequency_lookup = frequency_lookup[bug_report_id]

        class_with_packages_and_directory = file_index.replace('/', '.')

        f1 = feature_1(vectorized_report, vectorized_data, source_index, method_source_start_index, method_source_end_index)
        f2 = feature_2(enriched_report, enriched_api_data, enriched_api_lookup, file_index)
        f3 = feature_3(feature_3_data, current_feature_3_report_lookup, file_index)
        f4 = feature_4(current_bug_report_summary, file_index)

        f5 = 0.0
        for recency_file in current_recency_lookup.keys():
            if class_with_packages_and_directory.endswith(recency_file):
                f5 = current_recency_lookup[recency_file]
                break
        f6 = 0.0
        for frequency_file in current_frequency_lookup.keys():
            if class_with_packages_and_directory.endswith(frequency_file):
                f6 = current_recency_lookup[frequency_file]
                break

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

        used_in_fix = 0.0
        for result in bug_report['bug_report']['result']:
            if class_with_packages_and_directory.endswith(result):
                used_in_fix = 1.0
                break

        features.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, used_in_fix])
        features_files.append(file_index)

    sparse_features = sparse.csr_matrix(features)
    sparse.save_npz(data_prefix+'_'+bug_report_id+'_features', sparse_features)
    with open(data_prefix+'_'+bug_report_id+'_files', 'w') as outfile:
        json.dump(features_files, outfile)


if __name__ == '__main__':
    main()
