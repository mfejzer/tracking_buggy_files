#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports_without_desctiption.json> <bug_reports.json> <api_key> <url>
"""
import bugzilla
import datetime
import json
import time
import sys
from timeit import default_timer
from tqdm import tqdm


def main():
    print("Start", datetime.datetime.now().isoformat())
    before = default_timer()
    bug_reports_without_description_file_path = sys.argv[1]
    print("bug report without description file path", bug_reports_without_description_file_path)
    bug_reports_with_description_file_path = sys.argv[2]
    print("output bug report file path", bug_reports_with_description_file_path)
    api_key = sys.argv[3]
    url = sys.argv[4] # 'https://bugs.eclipse.org/bugs/rest/'

    add_missing_descriptions(bug_reports_without_description_file_path, bug_reports_with_description_file_path, api_key, url)

    after = default_timer()
    total = after - before
    print("End", datetime.datetime.now().isoformat())
    print("total time ", total)


def add_missing_descriptions(in_reports, out_reports, api_key, url):
    with open(in_reports) as bug_report_file:
        bug_reports = json.load(bug_report_file)

    empty_descriptions_keys = []
    for key in bug_reports:
        if bug_reports[key]['bug_report']['description'] is None:
            empty_descriptions_keys.append(key)
    print("Missing bug report keys size", len(empty_descriptions_keys))

    new_bug_reports = {}
    b = bugzilla.Bugzilla(url=url, api_key=api_key)
    for key in tqdm(empty_descriptions_keys):
        current_bug_id = bug_reports[key]['bug_report']['bug_id']
        try:
            time.sleep(1)
            comments = b.get_comments(current_bug_id)
            description = find_description(comments['bugs'][current_bug_id]['comments'])
            bug_reports[key]['bug_report']['description'] = description
            new_bug_reports[key] = bug_reports[key]
        except json.decoder.JSONDecodeError:
            print(current_bug_id)
            break

    with open(out_reports, 'w') as bug_report_out_file:
        json.dump(new_bug_reports, bug_report_out_file)


def find_description(comments):
    for comment in comments:
        if comment['count'] == 0:
            return comment['text']
    return ''

if __name__ == "__main__":
    main()
