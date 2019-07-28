#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from json import dump
from tqdm import tqdm


def main():
    bug_reports_file = sys.argv[1]
    converted_bug_reports_file = sys.argv[2]

    load_dataset(bug_reports_file, converted_bug_reports_file)


def load_dataset(bug_reports_file, converted_bug_reports_file):
    dataset = {}
    tree = ET.parse(bug_reports_file)
    bugrepository = tree.getroot()
    # find_fixing_commit(repository, "28974")
    # exit(0)
    for bug in tqdm(bugrepository.findall('bug')):
        id = bug.attrib['id']
        fixdate = bug.attrib['fixdate']
        timestamp = int(time.mktime(datetime.datetime.strptime(fixdate, "%Y-%m-%d %H:%M:%S").timetuple()))
        opendate = bug.attrib['opendate']
        open_timestamp = int(time.mktime(datetime.datetime.strptime(opendate, "%Y-%m-%d %H:%M:%S").timetuple()))
        buginformation = bug.find('buginformation')
        summary = buginformation.find('summary').text
        description = buginformation.find('description').text

        fixed_files = bug.find('fixedFiles')
        result = []
        for fix_file in fixed_files.findall('file'):
            result.append(fix_file.text)

        bug_report_content = {}
        bug_report_content['id'] = id
        bug_report_content['bug_id'] = id
        bug_report_content['summary'] = summary
        bug_report_content['timestamp'] = timestamp
        bug_report_content['open_timestamp'] = open_timestamp
        bug_report_content['status'] = 'closed'
        bug_report_content['result'] = result
        bug_report_content['description'] = description

        dataset[id] = {'bug_report': bug_report_content}

    with open(converted_bug_reports_file, 'w') as outfile:
        dump(dataset, outfile)


if __name__ == "__main__":
    main()
