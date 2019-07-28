#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <source_snapshot_file> <data_prefix>
"""

import json
from timeit import default_timer

import datetime
import numpy as np
import sys
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()
    source_snapshot_file = sys.argv[1]
    print("source snapshot file", source_snapshot_file)
    data_prefix = sys.argv[2]
    print("data_prefix", data_prefix)

    source_snapshot = load_source_snapshot_file(source_snapshot_file)

    extract_enriched_api(data_prefix, source_snapshot)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time", total)


def load_source_snapshot_file(soruce_snapshot_file_path):
    with open(soruce_snapshot_file_path, 'r') as infile:
        source_snapshot = json.load(infile)
    return source_snapshot


def load_file_index_lookup(data_prefix):
    with open(data_prefix + '_file_index_lookup', 'r') as infile:
        file_index_lookup = json.load(infile)
    return file_index_lookup


def load_data(data_prefix):
    data = sparse.load_npz(data_prefix + '_raw_count_data_before_tf_idf.npz').tocsr()
    return data


def load_types_lookup(data_prefix):
    with open(data_prefix + '_file_ast_types_lookup', 'r') as infile:
        types_lookup = json.load(infile)
    return types_lookup


def extract_enriched_api(data_prefix, source_snapshot):
    file_index_lookup = load_file_index_lookup(data_prefix)
    data = load_data(data_prefix)
    types_lookup = load_types_lookup(data_prefix)

    enriched_data = []
    api_lookup = {}

    n_rows = 0

    class_name_lookup = {}
    for file_index in types_lookup:
        for class_name in types_lookup[file_index]['classNames']:
            class_name_lookup[class_name] = file_index

    for file_index in tqdm(source_snapshot):
        ast_data, lookup, current_file = add_types_source_to_bug_report_data(data,
                                                                             file_index_lookup,
                                                                             types_lookup,
                                                                             class_name_lookup,
                                                                             file_index)
        current_index = n_rows
        enriched_data.append(ast_data)
        for k in lookup:
            lookup[k] += current_index
        api_lookup[current_file] = lookup
        n_rows += ast_data.shape[0]

    bug_report_rows, bug_report_lookup = get_bug_reports(data_prefix, data)
    enriched_data.extend(bug_report_rows)
    for bug_report_index in bug_report_lookup:
        bug_report_lookup[bug_report_index] += n_rows

    bug_report_data_matrix = sparse.vstack(enriched_data)

    sparse.save_npz(data_prefix + '_partial_enriched_api', bug_report_data_matrix)
    with open(data_prefix + '_partial_enriched_api_index_lookup', 'w') as outfile:
        json.dump(api_lookup, outfile)
    with open(data_prefix + '_partial_enriched_api_bug_report_index_lookup', 'w') as outfile:
        json.dump(bug_report_lookup, outfile)

    transformer = TfidfTransformer()
    tf_idf_data = transformer.fit_transform(bug_report_data_matrix)
    sparse.save_npz(data_prefix + '_tfidf_enriched_api', tf_idf_data)


def get_bug_reports(data_prefix, vectorized_data):
    with open(data_prefix + '_bug_report_index_lookup', 'r') as infile:
        bug_report_index_lookup = json.load(infile)
    indexes = []
    lookup = {}
    current_index = 0
    for bug_report_index in bug_report_index_lookup:
        indexes.append(bug_report_index_lookup[bug_report_index]['report'])
        lookup[bug_report_index] = current_index
        current_index += 1
    return vectorized_data[indexes, :], lookup


def add_types_source_to_bug_report_data(data, file_index_lookup, types, class_name_lookup, file_index):
    types_per_method = types[file_index]['methodVariableTypes']

    cl = data.shape[1]

    current_index = 0

    start = current_index
    enriched_apis = []
    for method_types in types_per_method:
        method_type_file_indexes = []

        for method_type in method_types:
            if method_type in class_name_lookup:
                method_type_file_indexes.append(class_name_lookup[method_type])

        supertypes_file_indexes_per_type = [set(find_types_file_indexes(types, class_name_lookup, m_t_f_i)) for m_t_f_i
                                            in method_type_file_indexes]

        indexes = []
        for supertypes in supertypes_file_indexes_per_type:
            indexes.extend(get_source_indexes(file_index_lookup, supertypes))

        if indexes == []:
            method_enriched_api = sparse.coo_matrix(np.zeros(cl).reshape(1, cl))
        else:
            method_enriched_api = sparse.coo_matrix(np.sum((data[indexes, :]), axis=0))

        enriched_apis.append(method_enriched_api)

    if enriched_apis == []:
        class_enriched_api = sparse.coo_matrix(np.zeros(cl).reshape(1, cl))
    else:
        class_enriched_api = sparse.coo_matrix(np.sum(enriched_apis, axis=0))

    enriched_apis.append(class_enriched_api)

    current_index += len(enriched_apis)

    lookup = {}
    lookup['enrichedApiStart'] = start
    lookup['enrichedApiEnd'] = current_index - 1

    enriched_apis_matrix = sparse.vstack(enriched_apis)

    return (enriched_apis_matrix, lookup, file_index)


def get_source_indexes(file_index_lookup, file_indexes):
    indexes = []
    for file_index in file_indexes:
        source_index = file_index_lookup[file_index]['source']
        indexes.append(source_index)
    return indexes


def find_types_file_indexes(types, class_name_lookup, file_index):
    result = []
    to_check = [file_index]
    while to_check:
        current_file_index = to_check.pop(0)
        if current_file_index not in result:
            result.append(current_file_index)
            supertypes = find_supertype_file_indexes(types, class_name_lookup, current_file_index)
            to_check.extend(supertypes)
    return result


def find_supertype_file_indexes(types, class_name_lookup, variable_file_index):
    if variable_file_index not in types:
        return []

    variable_type = types[variable_file_index]

    file_indexes = []
    for name in variable_type['superclassNames']:
        if name in class_name_lookup:
            file_indexes.append(class_name_lookup[name])
    for name in variable_type['interfaceNames']:
        if name in class_name_lookup:
            file_indexes.append(class_name_lookup[name])
    return file_indexes


if __name__ == '__main__':
    main()
