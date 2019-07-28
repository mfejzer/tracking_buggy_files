#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.json> <data_prefix>
"""

import json
from timeit import default_timer

import datetime
import sys
from collections import defaultdict
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
    report_open_date = {}
    result = {}
    for bug_report_id in tqdm(bug_reports):
        current_bug_report = bug_reports[bug_report_id]['bug_report']
        open_timestamp = current_bug_report['open_timestamp']
        report_open_date[bug_report_id] = open_timestamp
        modified_before = defaultdict(list)
        for other_bug_report_id in bug_reports:
            if other_bug_report_id != bug_report_id:
                other_bug_report = bug_reports[other_bug_report_id]['bug_report']
                other_close_timestamp = other_bug_report['timestamp']
                for fixed_file in other_bug_report['result']:
                    if fixed_file in current_bug_report['result']:
                        if datetime.date.fromtimestamp(other_close_timestamp) < datetime.date.fromtimestamp(
                                open_timestamp):
                            modified_before[fixed_file].append(other_close_timestamp)
        result[bug_report_id] = modified_before

    recency_lookup = {}
    frequency_lookup = {}
    for bug_report_id in result:
        modified_before = result[bug_report_id]
        open_timestamp = report_open_date[bug_report_id]
        open_month = datetime.datetime.fromtimestamp(open_timestamp).month
        current_fixes_recency = {}
        current_fixes_frequency = {}
        for file in modified_before:
            last_modification = max(modified_before[file])
            last_modification_month = datetime.date.fromtimestamp(last_modification).month
            d = (open_month - last_modification_month + 1.0)
            if d <= 0:
                recency = 0
            else:
                recency = (open_month - last_modification_month + 1.0) ** (-1)
            current_fixes_recency[file] = recency
            current_fixes_frequency[file] = len(modified_before[file])
        recency_lookup[bug_report_id] = current_fixes_recency
        frequency_lookup[bug_report_id] = current_fixes_frequency

    with open(data_prefix + '_feature_5_report_lookup', 'w') as outfile:
        json.dump(recency_lookup, outfile)
    with open(data_prefix + '_feature_6_report_lookup', 'w') as outfile:
        json.dump(frequency_lookup, outfile)


if __name__ == '__main__':
    main()
