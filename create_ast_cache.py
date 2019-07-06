#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <repository_path> <bug_reports.json> <data_prefix>
"""

import datetime
import json
import pickle
import subprocess
import sys

from joblib import Parallel, delayed

from date_utils import convert_commit_date
from multiprocessing import Pool
from operator import itemgetter
from timeit import default_timer
from tqdm import tqdm
from unqlite import UnQLite


def main():
    print("Start", datetime.datetime.now().isoformat()) 
    before = default_timer()
    repository_path = sys.argv[1]
    print("repository path", repository_path)
    bug_report_file_path = sys.argv[2]
    print("bug report file path", bug_report_file_path)
    data_prefix = sys.argv[3]
    print("data prefix", data_prefix)

    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        process(bug_reports, repository_path, data_prefix)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat()) 
    print("total time ", total)


def process(bug_reports, repository_path, data_prefix):
    ast_cache = prepare_ast_cache(repository_path)

    ast_cache_collection_db = UnQLite(data_prefix+"_ast_cache_collection_db")
  
    before = default_timer()
    for k, v in ast_cache.items():
        ast_cache_collection_db[k] = pickle.dumps(v, -1)
    after = default_timer()
    total = after - before
    print("total ast cache saving time ", total)

    bug_report_files = prepare_bug_report_files(repository_path, bug_reports, ast_cache)

    before = default_timer()

    bug_report_files_collection_db = UnQLite(data_prefix+"_bug_report_files_collection_db")
    for k, v in bug_report_files.items():
        bug_report_files_collection_db[k] = pickle.dumps(v, -1)
    after = default_timer()
    total = after - before
    print("total bug report files saving time ", total)


def list_notes(repository_path, refs = 'refs/notes/commits'):
    cmd = ['git', '-C', repository_path, 'notes', '--ref', refs, 'list']
    notes_list_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return [l.rstrip() for l in notes_list_process.stdout.readlines()]

def cat_file_blob(repository_path, sha, encoding='latin-1'):
    cmd = ['git', '-C', repository_path, 'cat-file', 'blob', sha]
    cat_file_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    result  = cat_file_process.stdout.read().decode(encoding)
    return result

def ls_tree(repository_path, sha):
    cmd = ['git', '-C', repository_path, 'ls-tree', '-r', sha]
    ls_tree_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return [l.rstrip() for l in ls_tree_process.stdout.readlines()]

def _process_notes(note, repository_path):
    (note_content_sha, note_object_sha) = str(note, 'utf-8').split(' ')

    note_content = cat_file_blob(repository_path, note_content_sha)
    ast_extraction_result = json.loads(note_content)
    return (note_object_sha, ast_extraction_result)

   
def _f(args):
    return _process_notes(args[0], args[1])


def prepare_ast_cache(repository_path):
    tokenized_refs = 'refs/notes/tokenized_counters'
    ast_notes = list_notes(repository_path, refs=tokenized_refs)
    print("existing tokenized notes ", len(ast_notes))

    before = default_timer()

    work = []
    for note in ast_notes:
        work.append((note, repository_path))
    pool = Pool(12, maxtasksperchild=1)
    ast_cache = dict(tqdm(pool.imap(_f, work), total=len(work)))

#    r = Parallel(n_jobs=6*12)(delayed(__process_notes)(i, repository_path) for i in tqdm(ast_notes)) 
#    ast_cache = dict(r)

    after = default_timer()
    total = after - before

    print("total ast cache retrieval time ", total)
    print("size of ast cache ", sys.getsizeof(ast_cache))
    return ast_cache

def sort_bug_reports_by_commit_date(bug_reports):
    commit_dates = []
    for index, commit in enumerate(tqdm(bug_reports)):
        sha = bug_reports[commit]['commit']['metadata']['sha'].replace('commit ','').strip()
        commit_date = convert_commit_date(bug_reports[commit]['commit']['metadata']['date'].replace('Date:','').strip())
        commit_dates.append((sha, commit_date))

    sorted_commit_dates = sorted(commit_dates, key=itemgetter(1))
    sorted_commits = [commit_date[0] for commit_date in sorted_commit_dates]
    return sorted_commits

def _load_parent_commit_files(repository_path, commit, ast_cache):
    parent = commit+'^'

    class_name_to_sha = {}
    sha_to_file_name = {}
    shas = []
    for ls_entry in ls_tree(repository_path, parent):
        (file_sha_part, file_name) = str(ls_entry,'utf-8').split('\t')
        file_sha = file_sha_part.split(' ')[2]
        #file_sha = intern(file_sha)
        #file_name = intern(file_name)
        if file_name.endswith(".java") and file_sha in ast_cache:
            #shas.append(intern(file_sha))
            file_sha_ascii = file_sha
            shas.append(file_sha_ascii)
            class_names = ast_cache[file_sha]['classNames']
            for class_name in class_names:
                class_name_ascii = class_name
                class_name_to_sha[class_name_ascii] = file_sha_ascii
            sha_to_file_name[file_sha_ascii] = file_name

    f_lookup = {}
    f_lookup['shas'] = shas
    f_lookup['class_name_to_sha'] = class_name_to_sha
    f_lookup['sha_to_file_name'] = sha_to_file_name
    return (commit.encode('ascii','ignore'), f_lookup)


def prepare_bug_report_files(repository_path, bug_reports, ast_cache):
    sorted_commits = sort_bug_reports_by_commit_date(bug_reports)

    before = default_timer()

    r = Parallel(n_jobs=6*12,backend="threading")(delayed(_load_parent_commit_files)(repository_path, commit, ast_cache) for commit in tqdm(sorted_commits)) 
    bug_report_files = dict(r)

    after = default_timer()
    total = after - before
    print("total bug report files retrieval time ", total)
    return bug_report_files

if __name__ == '__main__':
    main()
