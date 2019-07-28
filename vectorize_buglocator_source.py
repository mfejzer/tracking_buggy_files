#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <tokenized_source.json> <data_prefix>
Requires results of:
process_buglocator.py
tokenize_buglocator_source.py
"""

import json
from timeit import default_timer

import datetime
import sys
from collections import Counter
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from operator import itemgetter
from re import finditer
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from date_utils import convert_commit_date


def main():
    print("Start", datetime.datetime.now().isoformat()) 
    before = default_timer()
    bug_report_file_path = sys.argv[1]
    print("bug report file path", bug_report_file_path)
    tokenized_source_snapshot_file = sys.argv[2]
    print("tokenized source snapshot file", tokenized_source_snapshot_file)
    data_prefix = sys.argv[3]
    print("data_prefix", data_prefix)
    bug_reports = load_bug_reports(bug_report_file_path)

    vectorize(tokenized_source_snapshot_file, bug_reports, data_prefix)
    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat()) 
    print("total time ", total)


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


def vectorize(tokenized_source_snapshot_file, bug_reports, data_prefix):
    data = []
    file_index_lookup = {}
    ast_types_lookup = {}
    current_index = 0

    with open(tokenized_source_snapshot_file, 'r') as infile:
        source_snapshot = json.load(infile)

    for file in tqdm(source_snapshot):
        source = source_snapshot[file]['tokenized_sources']
        graph = source_snapshot[file]['graph']
        data, current_index, current_lookup = add_ast_to_vectorization_data(data, current_index, source)
        file_index_lookup[file] = current_lookup
        ast_types_lookup[file] = extract_types(source)

    stemmer = PorterStemmer()

    print("data length", len(data))
    print("current index", current_index)
    
    bug_report_index_lookup = {}
    for bug_report_id in tqdm(bug_reports):
        current_bug_report = bug_reports[bug_report_id]['bug_report']
        data, current_index, current_lookup = add_bug_report_to_vectorization_data(data, current_index, current_bug_report, stemmer)
        bug_report_index_lookup[bug_report_id] = current_lookup

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
    sparse.save_npz(data_prefix+'_raw_count_data_before_tf_idf', vectorized_data)

    before_tf_idf = default_timer()
    transformer = TfidfTransformer()
    tf_idf_data = transformer.fit_transform(vectorized_data)
    after_tf_idf = default_timer()
    total_tf_idf = after_tf_idf - before_tf_idf
    print("total count tf idf time ", total_tf_idf)
    print("tf_idf_data type ", type(tf_idf_data))
    print("tf_idf_data shape", tf_idf_data.shape)

    feature_names = vectorizer.get_feature_names()
    feature_names_lenghts_dict = {}
    for i, feature_name in enumerate(feature_names):
        feature_names_lenghts_dict[i] = len(feature_name)
    with open(data_prefix+'_feature_names_dict', 'w') as outfile:
        json.dump(feature_names_lenghts_dict, outfile)
    with open(data_prefix+'_file_index_lookup', 'w') as outfile:
        json.dump(file_index_lookup, outfile)
    with open(data_prefix+'_file_ast_types_lookup', 'w') as outfile:
        json.dump(ast_types_lookup, outfile)
    with open(data_prefix+'_bug_report_index_lookup', 'w') as outfile:
        json.dump(bug_report_index_lookup, outfile)
    sparse.save_npz(data_prefix+'_raw_count_data', tf_idf_data)


if __name__ == '__main__':
    main()
