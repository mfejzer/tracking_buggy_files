#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <data_prefix>
"""

import json
from timeit import default_timer

import datetime
import sys
from collections import Counter
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from re import finditer
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()

    bug_report_file_path = sys.argv[1]
    print("bug report file path", bug_report_file_path)
    data_prefix = sys.argv[2]
    print("data prefix", data_prefix)

    bug_reports = load_bug_reports(bug_report_file_path)

    process(data_prefix, bug_reports)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time", total)


def load_bug_reports(bug_report_file_path):
    """load bug report file (the one generated from xml)"""
    with open(bug_report_file_path) as bug_report_file:
        bug_reports = json.load(bug_report_file)
        return bug_reports


def process(data_prefix, bug_reports):
    stemmer = PorterStemmer()

    feature_3_data_list = []
    feature_3_report_lookup = {}

    current_index = 0
    for bug_report_id in tqdm(bug_reports):
        current_bug_report = bug_reports[bug_report_id]['bug_report']
        open_timestamp = current_bug_report['open_timestamp']
        fixed_files = current_bug_report['result']

        lookup = {}

        report = str(current_bug_report['summary']) + str(current_bug_report['description'])
        report_tokens = tokenize(report, stemmer)
        feature_3_data_list.append(report_tokens)
        lookup['report'] = current_index
        current_index += 1

        file_looup = {}
        for fixed_file in fixed_files:
            combined_reports = combine_reports_fixing_same_file_before_date(bug_reports, fixed_file, open_timestamp)
            combined_report_tokens = tokenize(combined_reports, stemmer)
            feature_3_data_list.append(combined_report_tokens)
            file_looup[fixed_file] = current_index
            current_index += 1
        lookup['files'] = file_looup
        feature_3_report_lookup[bug_report_id] = lookup

    before_v = default_timer()
    vectorizer = DictVectorizer()
    vectorized_data = vectorizer.fit_transform(feature_3_data_list)
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

    sparse.save_npz(data_prefix + '_feature_3_data', tf_idf_data)
    with open(data_prefix + '_feature_3_report_lookup', 'w') as outfile:
        json.dump(feature_3_report_lookup, outfile)


def combine_reports_fixing_same_file_before_date(bug_reports, selected_file, open_timestamp):
    summaries = []
    for bug_report_id in bug_reports:
        current_bug_report = bug_reports[bug_report_id]['bug_report']
        if selected_file in current_bug_report['result'] and current_bug_report['timestamp'] < open_timestamp:
            summaries.append(str(current_bug_report['summary']))
            # summaries.append(str(current_bug_report['summary'] + str(current_bug_report['description']))

    return ' '.join(summaries)


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


if __name__ == '__main__':
    main()
