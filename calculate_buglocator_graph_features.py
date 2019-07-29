#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <tokenized_source_snapshot_file> <data_prefix>
Requires results of "java-ast-extractor-graph-notes.jar" in repository
"""

import collections as col
import json
from timeit import default_timer

import datetime
import networkx as nx
import sys
from scipy import sparse

from project_import_graph_features import process_graph


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()
    tokenized_source_snapshot_file = sys.argv[1]
    print("tokenized source snapshot file", tokenized_source_snapshot_file)
    data_prefix = sys.argv[2]
    print("data_prefix", data_prefix)

    soruce_snapshot = load_source_snapshot_file(tokenized_source_snapshot_file)

    calculate_graph_features(data_prefix, soruce_snapshot)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time ", total)


def load_source_snapshot_file(soruce_snapshot_file_path):
    with open(soruce_snapshot_file_path, 'r') as infile:
        source_snapshot = json.load(infile)
    return source_snapshot


def calculate_graph_features(data_prefix, soruce_snapshot):

    graph_features_data_list = []
    graph_features_lookup = {}

    file_to_imports = {}
    file_to_class_name = {}
    for file_index in soruce_snapshot:
        imports = soruce_snapshot[file_index]['graph']

        file_to_imports[file_index] = imports
        if 'className' in imports and imports['className'] is not None and imports['className'] != "":
            class_name = imports['className']
            class_name = class_name.replace(".", "")
            file_to_class_name[file_index] = class_name

    graph_data = process_graph_results(file_to_imports)

    current_index = 0
    for file_index in soruce_snapshot:
        try:
            current_node_name = file_to_class_name[file_index]
            # print(current_node_name)
            # print(graph_data.loc[current_node_name])
            # exit(0)
            values = graph_data.loc[current_node_name]
            feature_15 = values['in']
            feature_16 = values['out']
            feature_17 = values['pr']
            feature_18 = values['a']
            feature_19 = values['h']
        except KeyError:
            feature_15 = 0.0
            feature_16 = 0.0
            feature_17 = 0.0
            feature_18 = 0.0
            feature_19 = 0.0

        current_features = sparse.coo_matrix([feature_15, feature_16, feature_17, feature_18, feature_19])
        #        print(current_features.shape)
        #        print(current_features)
        #        exit(0)
        graph_features_data_list.append(current_features)
        graph_features_lookup[file_index] = current_index
        current_index += 1

    graph_features_data = sparse.vstack(graph_features_data_list)

    sparse.save_npz(data_prefix + '_graph_features_data', graph_features_data)
    with open(data_prefix + '_graph_features_index_lookup', 'w') as outfile:
        json.dump(graph_features_lookup, outfile)


def process_graph_results(file_to_imports):
    graph_preparation = col.defaultdict(list)
    class_names_to_paths = {}

    __c = 0
    __d = 0
    for file, class_details in file_to_imports.items():
        __d += 1
        try:
            class_name = class_details['className']
            class_name = class_name.replace(".", "")

        except Exception:
            __c += 1
            continue
        class_names_to_paths[class_name] = class_details['fileName']
        dependencies = class_details['dependencies']
        for dependency, _ in dependencies.items():
            graph_preparation[class_name].append(dependency.replace(".", ""))
        if len(graph_preparation[class_name]) == 0:
            #print("AAAAAAAAAAAA", class_name)
            del graph_preparation[class_name]

    # print(graph_preparation)
    G = nx.from_dict_of_lists(graph_preparation, nx.DiGraph())

    # print(__c / __d)
    # exit()

    # print(G)
    #print("Nodes", G.number_of_nodes())
    #print("Edges", G.degree(weight='weight'))
    # print(len(list(G.nodes())))
    # print(list(G.nodes()))
    #exit()
    return process_graph(G)


if __name__ == '__main__':
    main()
