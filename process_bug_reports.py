#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(scriptName) <bug_reports.xml> <repository_path> <bug_reports.json>
"""

import collections 
import itertools
import psutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from json import dumps
from tqdm import tqdm

def main():
    bug_reports_file = sys.argv[1]
    repository = sys.argv[2]
    json_file_name = sys.argv[3]
    dataset = load_dataset(bug_reports_file, repository)
    with open(json_file_name, 'w') as f:
        json.dump(dataset, f)
    # print(dumps(dataset))


def load_dataset(bug_reports_file, repository):
    dataset = {} 
    tree = ET.parse(bug_reports_file)
    root = tree.getroot()
    for database in root.findall('database'):
        for bug_report in tqdm(database.findall('table')):
            fixing_commit = bug_report.find("column[@name='commit']").text
            bug_report_content = {}
            bug_report_content['commit'] = fixing_commit
            bug_report_content['id'] = bug_report.find("column[@name='id']").text
            bug_report_content['bug_id'] = bug_report.find("column[@name='bug_id']").text
            bug_report_content['summary'] = bug_report.find("column[@name='summary']").text
            bug_report_content['timestamp'] = bug_report.find("column[@name='report_timestamp']").text
            bug_report_content['status'] = bug_report.find("column[@name='status']").text
            bug_report_content['result'] = bug_report.find("column[@name='result']").text
            bug_report_content['description'] = bug_report.find("column[@name='description']").text
             
            commit_content = retrieve_commit(repository, fixing_commit)
           
            dataset[fixing_commit] = {'bug_report':bug_report_content, 'commit':commit_content}

    return dataset

#    with open(fixes_file) as fixes_file_handle:
#        for line in tqdm(fixes_file_handle):
#            fix = line.decode('latin-1').rstrip()
#            dataset[fix] = retrieve_commit(repository, fix)
#    return dataset

def retrieve_commit(repository, commit, ext='.java'):
    metadata = retrieve_metadata(repository, commit)
    diff = retrieve_diff(repository, commit, ext=ext)
    commit_content = {}
    commit_content['metadata'] = metadata
    commit_content['diff'] = diff
    return commit_content

def retrieve_metadata(repository, commit):
    full_sha = None
    author = None
    date = None
    message = ''
    cmd = 'git -C ' + repository + ' show -s ' + commit
#    print cmd
    process = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    for index, line in enumerate(iter(process.stdout.readline, '')):
        line = line.decode('latin-1')
        if index == 0:
             full_sha = line
        elif index == 1:
             author = line
        elif index == 2:
             date = line
        else:
             message += line
    metadata = {}
    metadata['sha'] = full_sha
    metadata['author'] = author
    metadata['date'] = date
    metadata['message'] = message
    return metadata
        

def retrieve_diff(repository, commit, ext='.java'):
    cmd = 'git -C ' + repository + ' diff-tree --no-commit-id --name-only -r ' + commit
#    print cmd
    files = {}
    process = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    for line in iter(process.stdout.readline, ''):
        filepath = line.decode('latin-1').rstrip()
#        print filepath
        if filepath != '' and filepath.endswith(ext):
            files[filepath] = retrieve_diff_on_filepath(repository, commit, filepath)
    return files

def retrieve_diff_on_filepath(repository, commit, filepath):
    cmd = 'git -C ' + repository + ' diff --unified=0 --no-prefix ' + commit + '^ '+commit +' -- '+filepath
#    print cmd
    diff_lines = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE).stdout.read().decode('latin-1')
#    print diff_lines
    return diff_lines
 
if __name__ == "__main__":
    main()


