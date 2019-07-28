#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) source_snapshot_file

Requires results of "java-ast-extractor-source-snapshot.jar"
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
from tqdm import tqdm


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()
    source_snapshot_file = sys.argv[1]
    tokenized_source_snapshot_file = sys.argv[2]

    process(source_snapshot_file, tokenized_source_snapshot_file)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time %f" % total)


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


def process(source_snapshot_file, tokenized_source_snapshot_file):
    stemmer = PorterStemmer()

    with open(source_snapshot_file, 'r') as infile:
        source_snapshot = json.load(infile)

    result_snapshot = {}
    for file in tqdm(source_snapshot):
        source = source_snapshot[file]['source']
        graph = source_snapshot[file]['graph']

        tokenized_sources = tokenize_source(source, stemmer)

        result_snapshot[file] = {'tokenized_sources': tokenized_sources, 'graph': graph}

    with open(tokenized_source_snapshot_file, 'w') as outfile:
        json.dump(result_snapshot, outfile)

 
def tokenize_source(ast_extraction_result, stemmer):
    tokenized_source = tokenize(ast_extraction_result['rawSourceContent'], stemmer)
    tokenized_methods = [tokenize(method, stemmer) for method in ast_extraction_result['methodContent']]

    tokenized_superclass_names = [tokenize(name, stemmer) for name in ast_extraction_result['superclassNames']]
    tokenized_interface_names = [tokenize(name, stemmer) for name in ast_extraction_result['interfaceNames']]
    tokenized_class_names = [tokenize(name, stemmer) for name in ast_extraction_result['classNames']]
    tokenized_method_names = [tokenize(name, stemmer) for name in ast_extraction_result['methodNames']]
    tokenized_variable_names = [tokenize(name, stemmer) for name in ast_extraction_result['variableNames']]
    tokenized_comments = [tokenize(comment, stemmer) for comment in ast_extraction_result['commentContent']]

    tokenized_ast = {}
    tokenized_ast['tokenizedSource'] = tokenized_source
    tokenized_ast['tokenizedMethods'] = tokenized_methods

    tokenized_ast['tokenizedSuperclassNames'] = tokenized_superclass_names
    tokenized_ast['tokenizedInterfaceNames'] = tokenized_interface_names
    tokenized_ast['tokenizedClassNames'] = tokenized_class_names
    tokenized_ast['tokenizedMethodNames'] = tokenized_method_names
    tokenized_ast['tokenizedVariableNames'] = tokenized_variable_names
    tokenized_ast['tokenizedComments'] = tokenized_comments

    tokenized_ast['classNames'] = ast_extraction_result['classNames']
    tokenized_ast['superclassNames'] = ast_extraction_result['superclassNames']
    tokenized_ast['interfaceNames'] = ast_extraction_result['interfaceNames']
    tokenized_ast['methodVariableTypes'] = ast_extraction_result['methodVariableTypes']

    return tokenized_ast


if __name__ == '__main__':
    main()
