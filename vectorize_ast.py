#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <data_prefix>
"""

import json
from timeit import default_timer

import datetime
import pickle
import sys
from collections import Counter
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from operator import itemgetter
from re import finditer
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
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

    ast_cache_db = UnQLite(data_prefix+"_ast_cache_collection_db")

    vectorize(ast_cache_db, bug_reports, data_prefix)
    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat()) 
    print("total time ", total)
    ast_cache_db.close()


def load_bug_reports(bug_report_file_path):
    """load bug report file (the one generated from xml)"""
    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        return bug_reports


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

removed = u'!"#%&\'()*+,-./:;<=>?@[\]^_`{|}~1234567890'
utf_translate_table = dict((ord(char), u' ') for char in removed)
stop_words = set(stopwords.words('english'))

def tokenize(text, stemmer):
    sanitized_text = text.translate(utf_translate_table)
    tokens = wordpunct_tokenize(sanitized_text)
    all_tokens = []
    for token in tokens:
        additional_tokens = camel_case_split(token)
        if len(additional_tokens)>1:
            for additional_token in additional_tokens:
                all_tokens.append(additional_token)
        all_tokens.append(token)
    return Counter([stemmer.stem(token) for token in all_tokens if token.lower() not in stop_words])


def sort_bug_reports_by_commit_date(bug_reports):
    commit_dates = []
    for index, commit in enumerate(tqdm(bug_reports)):
        sha = bug_reports[commit]['commit']['metadata']['sha'].replace('commit ','').strip()
        commit_date = convert_commit_date(bug_reports[commit]['commit']['metadata']['date'].replace('Date:','').strip())
        commit_dates.append((sha, commit_date))

    sorted_commit_dates = sorted(commit_dates, key=itemgetter(1))
    sorted_commits = [commit_date[0] for commit_date in sorted_commit_dates]
    return sorted_commits

def convert(dict_list):
    counter_list = []
    for d in dict_list:
        counter_list.append(Counter(d))
    return counter_list

def add_ast_to_vectorization_data(data, current_index, ast):
    """add ast to list of data for vectorization, create dictionary of indexes"""
    ast_dict = {}

    data.extend(convert([ast['tokenizedSource']]))
    ast_dict['source'] = current_index
    current_index += 1

    data.extend(convert(ast['tokenizedMethods']))
    ast_dict['methodsStart'] = current_index
    current_index += len(ast['tokenizedMethods'])
    ast_dict['methodsEnd'] = current_index - 1

    data.extend(convert(ast['tokenizedClassNames']))
    ast_dict['classNamesStart'] = current_index
    current_index += len(ast['tokenizedClassNames'])
    ast_dict['classNamesEnd'] = current_index - 1

    data.extend(convert(ast['tokenizedMethodNames']))
    ast_dict['methodNamesStart'] = current_index
    current_index += len(ast['tokenizedMethodNames'])
    ast_dict['methodNamesEnd'] = current_index - 1

    data.extend(convert(ast['tokenizedVariableNames']))
    ast_dict['variableNamesStart'] = current_index
    current_index += len(ast['tokenizedVariableNames'])
    ast_dict['variableNamesEnd'] = current_index - 1

    data.extend(convert(ast['tokenizedComments']))
    ast_dict['commentsStart'] = current_index
    current_index += len(ast['tokenizedComments'])
    ast_dict['commentsEnd'] = current_index - 1

    return (data, current_index, ast_dict)

def add_bug_report_to_vectorization_data(data, current_index, bug_report, stemmer):
    """add bug report to list of data for vectorization, create dictionary of indexes"""
    bug_report_dict = {}

    summary = bug_report['summary']
    if summary is None:
        summary = u''
    summary_tokens = tokenize(summary,stemmer)
    data.append(summary_tokens)
    bug_report_dict['summary'] = current_index
    current_index += 1
    
    description = bug_report['description']
    if description is None:
        description = u''
    description_tokens = tokenize(description,stemmer)
    data.append(description_tokens)
    bug_report_dict['description'] = current_index
    current_index += 1

    report = description + u' ' + summary
    report_tokens = tokenize(report,stemmer)
    data.append(report_tokens)
    bug_report_dict['report'] = current_index
    current_index += 1
   
    return (data, current_index, bug_report_dict)

def extract_types(ast):
    types = {}
    types['superclassNames'] = ast['superclassNames']
    types['interfaceNames'] = ast['interfaceNames']
    types['methodVariableTypes'] = ast['methodVariableTypes']
    types['classNames'] = ast['classNames']
    return types

def vectorize(ast_cache, bug_reports, data_prefix):
    data = []
    current_index = 0
    
    ast_index_lookup = {}
    ast_types_lookup = {}
    with ast_cache.cursor() as cursor:
        for k, v in cursor:
            ast_sha = k
            current_ast = pickle.loads(v)
            data, current_index, current_lookup = add_ast_to_vectorization_data(data, current_index, current_ast)
            ast_index_lookup[ast_sha] = current_lookup
            ast_types_lookup[ast_sha] = extract_types(current_ast)

            
    stemmer = PorterStemmer()

    print("data length", len(data))
    print("current index", current_index)
    
    bug_report_index_lookup = {}
    for bug_report_id in tqdm(bug_reports):
        current_bug_report = bug_reports[bug_report_id]['bug_report']
        data, current_index, current_lookup = add_bug_report_to_vectorization_data(data, current_index, current_bug_report, stemmer)
        bug_report_index_lookup[bug_report_id[0:7]] = current_lookup


    print("data length", len(data))
    print("current index", current_index)

    before_v = default_timer()
    vectorizer = DictVectorizer()
    vectorized_data = vectorizer.fit_transform(data)
    after_v = default_timer()
    total_v = after_v - before_v
    print("total count vectorization time ", total_v)
    print("vectorized_data type ", type(vectorized_data))
    print("vectorized_data shape", vectorized_data.shape)


    feature_names = vectorizer.get_feature_names()
    feature_names_lenghts_dict = {}
    for i, feature_name in enumerate(feature_names):
        feature_names_lenghts_dict[i] = len(feature_name)
    with open(data_prefix+'_feature_names_dict', 'w') as outfile:
        json.dump(feature_names_lenghts_dict, outfile)

    sparse.save_npz(data_prefix+'_raw_count_data', vectorized_data)

    ast_index_collection = UnQLite(data_prefix+"_ast_index_collection_index_db")
    for k, v in ast_index_lookup.items():
        ast_index_collection[k] = pickle.dumps(v,-1)

    bug_report_index_collection = UnQLite(data_prefix+"_bug_report_index_collection_index_db")
    for k, v in bug_report_index_lookup.items():
        bug_report_index_collection[k] = pickle.dumps(v,-1)

    ast_types_collection = UnQLite(data_prefix+"_ast_types_collection_index_db")
    for k, v in ast_types_lookup.items():
        ast_types_collection[k] = pickle.dumps(v,-1)

    ast_index_collection.close()
    bug_report_index_collection.close()
    ast_types_collection.close()

if __name__ == '__main__':
    main()
