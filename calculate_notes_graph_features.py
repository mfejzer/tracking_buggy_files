#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <data_prefix> <repository_path>
Requires results of "java-ast-extractor-graph-notes.jar" in repository
"""

import collections as col
import json
from timeit import default_timer

import datetime
import networkx as nx
import pickle
import subprocess
import sys
from multiprocessing import Manager, Pool
from operator import itemgetter
from scipy import sparse
from tqdm import tqdm
from unqlite import UnQLite

from date_utils import convert_commit_date
from project_import_graph_features import process_graph


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()
    bug_report_file_path = sys.argv[1]
    print("bug report file path", bug_report_file_path)
    data_prefix = sys.argv[2]
    print("data prefix", data_prefix)
    repository_path = sys.argv[3]
    print("repository path", repository_path)

    fixes_list = extract_fixes_list(bug_report_file_path)

    calculate_graph_features_for_each_fix(fixes_list, data_prefix, repository_path)

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


def calculate_graph_features_for_each_fix(fixes_list, data_prefix, repository_path):
    notes = list_notes(repository_path)
    sha_to_note = {}
    for note in notes:
        note_content_sha, note_object_sha = note.split(' ')
        sha_to_note[note_object_sha] = note_content_sha
    manager = Manager()
    d = manager.dict()
    d.update(sha_to_note)

    work = []
    for fixing_commit in fixes_list:
        work.append((data_prefix, fixing_commit[0], fixing_commit[1], repository_path, d))

    pool = Pool(12, maxtasksperchild=1)
    r = list(tqdm(pool.imap(_f, work), total=len(work)))
    # print(_f(work[0]))


def _f(args):
    return calculate_graph_features(args[0], args[1], args[2], args[3], args[4])


def calculate_graph_features(data_prefix, bug_report_id, bug_report_full_sha, repository_path, sha_to_note):
    bug_report_files_collection_db = UnQLite(data_prefix + "_bug_report_files_collection_db",
                                             flags=0x00000100 | 0x00000001)
    current_files = pickle.loads(bug_report_files_collection_db[bug_report_full_sha])
    bug_report_files_collection_db.close()

    shas = current_files['shas']
    sha_to_file_name = current_files['sha_to_file_name']

    graph_features_data_list = []
    graph_features_lookup = {}

    sha_to_imports = {}
    sha_to_class_name = {}
    for sha in shas:
        note_sha = sha_to_note[sha]
        note_content = cat_file_blob(repository_path, note_sha)
        imports = json.loads(note_content)
        sha_to_imports[sha] = imports
        if 'className' in imports and imports['className'] is not None and imports['className'] != "":
            class_name = imports['className']
            class_name = class_name.replace(".", "")
            sha_to_class_name[sha] = class_name

    graph_data = process_graph_results(sha_to_imports)

    current_index = 0
    for sha in shas:
        current_file_name = sha_to_file_name[sha]
        # print(sha)
        # print(current_file_name)
        try:
            current_node_name = sha_to_class_name[sha]
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
        graph_features_lookup[sha] = current_index
        current_index += 1

    graph_features_data = sparse.vstack(graph_features_data_list)

    sparse.save_npz(data_prefix + '_' + bug_report_id[0:7] + '_graph_features_data', graph_features_data)
    with open(data_prefix + '_' + bug_report_id[0:7] + '_graph_features_index_lookup', 'w') as outfile:
        json.dump(graph_features_lookup, outfile)

    return (bug_report_id)


def process_graph_results(sha_to_imports):
    graph_preparation = col.defaultdict(list)
    class_names_to_paths = {}

    __c = 0
    __d = 0
    for sha, class_details in sha_to_imports.items():
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


def list_notes(repository_path, refs='refs/notes/graph'):
    cmd = ' '.join(['git', '-C', repository_path, 'notes', '--ref', refs, 'list'])
    notes_lines = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read().decode('latin-1').split('\n')
    return notes_lines[:-1]


def cat_file_blob(repository_path, sha, encoding='latin-1'):
    cmd = ' '.join(['git', '-C', repository_path, 'cat-file', 'blob', sha])
    cat_file_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result = cat_file_process.stdout.read().decode(encoding)
    return result


if __name__ == '__main__':
    main()
