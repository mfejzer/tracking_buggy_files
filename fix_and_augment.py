#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Usage: %(prog)s [options] <json-file> <repository-path>

This script fixes the errors from adding data from repository by
process_bug_reports.py script to the data from bugtracker.  It also
adds the 'timestamp' filed to the commit metadata, if it does not
exist.  It makes it easier to compare the time bug report was created
(which 'timestamp' field in 'bug_report' is most probably about) with
the time it or the other bug report was fixed.

It also augments 'bug_report' data with 'preceding_commit' field (if
it does not exist yet) with better approximation for version that was
used by submitter when writing bug report than the commit before the
bugfix.  The script would simply find first commit older than the bug
report timestamp.  The value of this field is full identifier of said
commit (full identifier just in case).

It can optionally sort data by bug timestamp, or by timestamp of the
commit fixing the bug.  When not sorting, this script preserves the
order of keys in JSON.
"""


import sys
import re
import json
import argparse

from tqdm import tqdm
from collections import OrderedDict

from date_utils import convert_commit_date, datetime_to_timestamp
from dataset_utils import sorted_by_bugreport, sorted_by_commit
import args_utils
import git_utils


def main():
    ### run as script

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Normalize, fix and augment bug + repo data"+
                    ", augment repo")
    parser = args_utils.add_arguments_and_usage(parser, repo_required=True)
    parser = args_utils.add_json_output_options(parser)
    parser = args_utils.add_git_backend_selection(parser)
    parser.add_argument('--sort', choices=['bug','commit'],
                        nargs='?', const='bug',
                        help="sorting by bug or commit timestamp"+
                             " [default '%(const)s']")
    parser.add_argument('--sort-keys', action='store_true',
                        help="sort (sub)keys in predefined order")
    parser.add_argument('-t', '--add-tags', action='store_true',
                        help="add 'fixes-<bug_id>' tags in repo")
    args = parser.parse_args()

    # process arguments
    datafile  = args.json_file
    repo_path = args.repository_path
    repo = args_utils.repo_by_backend(repo_path, args.git_backend)

    # read data
    data = args_utils.read_json(datafile, preserve_order=True)

    # process data
    data = process_data(data, repo, **vars(args))

    # print or save results
    args_utils.print_json(data, args)


def process_data(data, repo,
                 sort='bug', sort_keys=False, add_tags=False,
                 **kwargs):
    """Fix / trim contents and augment data

    This is the major function of this script.  It cleans up contents,
    optionally sorts data and/or sort keys in data (sort nested
    structures), and augments it with information from data and from
    repository.

    It can also optionally augment repository with tags, for each
    bugfix commit denoting which bug it fixed.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects implementation used.
        One could use the result of args_utils.repo_by_backend()
        here

    sort : 'bug' | 'commit' | false, optional
        Specifies how to sort entries in data.  Unknown or false value
        turns off sorting.

    sort_keys : bool, optional
        Whether to sort keys in (nested structure of) data.

    add_tags : bool, optional
        Whether to add tags to the repository.

    **kwargs
        Arbitrary keyword arguments, to be ignored.  This allow
        passing "**vars(args)", where args=parser.parse_args(),
        to this function, ant it would ignore unknown keys/options.

    Returns
    -------
    dict | OrderedDict
        Fixed and augmented data.
    """
    fix_commit_metadata(data) # also ensures 'timestamp' for commit
    find_preceding_commits(data, repo)
    if sort and sort in ('bug', 'commit'):
        print('Sorting data by %s timestamp' % sort,
              file=sys.stderr)
    if sort == 'bug':
        data = sorted_by_bugreport(data, key='timestamp')
    elif sort == 'commit':
        data = sorted_by_commit(data, key='timestamp')

    if sort_keys:
        print('Sorting keys in data in predefined order',
              file=sys.stderr)
        data_sort_keys(data)

    # augment repository using tags
    if add_tags:
        tag_bugfixing_commits(repo, data)

    return data


def fix_commit_metadata(data):
    """Trim metadata fields, and add 'timestamp' from 'date'

    Assumes that if 'timestamp' exists, then there is no need
    for fixes.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from the
        JSON file.

    Returns
    -------
    dict | OrderedDict
        Changed and augmented data.

    Side effects
    ------------
    Changes its input.
    """
    n_skipped = 0
    for commit in tqdm(data):
#        if 'timestamp' in data[commit]['commit']['metadata']:
#            n_skipped = n_skipped + 1
#            continue

        trim_commit_info(data[commit]['commit']['metadata'])
        data[commit]['commit']['metadata']['timestamp'] = \
            datetime_to_timestamp(convert_commit_date(
                data[commit]['commit']['metadata']['date'].replace('Date:', '').strip()
            ))

    print('%d / %d skipped: had already "timestamp" field in commit metadata' %
          (n_skipped, len(data)),
          file=sys.stderr)

    return data


def find_preceding_commits(data, repo):
    """For each bug report, find commit just preceding its creation

    This found commit (first commit older than the bug report
    timestamp starting from the bugfix commit) is then stored in the
    'preceding_commit' field in 'bug_report' section.  If this field
    exists, it is assumed that it is correct, and the calculations
    skipped.

    The commit just preceding the creation of the bug report is meant
    to be an approximation of the state / version of the project that
    bug report author was using; the version in which te bug was
    found.  NOTE that Bugzilla bug tracker has 'Version' field, but
    this information was not present in the original dataset.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects implementation used.
        This could be the result of args_utils.repo_by_backend()

    Returns
    -------
    dict | OrderedDict
        Augmented data.

    Side effects
    ------------
    Changes its 'data' input.
    """
    n_skipped = 0
    field_name = 'preceding_commit'
    for commit in tqdm(data):
        if field_name in data[commit]['bug_report']:
            n_skipped = n_skipped + 1
            continue
        data[commit]['bug_report'][field_name] = \
            git_utils.find_commit_by_timestamp(repo,
                timestamp=data[commit]['bug_report']['timestamp'],
                # 'sha' requires fix_commit_metadata() first,
                # or simply we could use `start_commit=commit`
                start_commit=data[commit]['commit']['metadata']['sha'],
            )

    print('%d / %d skipped: had already "%s" field for bug report' %
          (n_skipped, len(data), field_name),
          file=sys.stderr)

    return data


def tag_bugfixing_commits(repo, data):
    """Add tags to commits which are bug fixes in the form of 'fixes-<bug_id>'

    For each bug report of a fixed bug in data, retrieve which commit
    fixed the bug in question.  To each such commit, add lightweight
    tag denoting which bug it fixes to the repository.  This could be
    later used to find out which bugfixes affected given file.

    NOTE that the script needs to have write permissions to the
    repository in question.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    repo :  str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects implementation used.
        This could be the result of args_utils.repo_by_backend()

    Side effects
    ------------
    Creates lightweight tags in the repository.  Prints progress
    messages on stderr.
    """
    n_skipped = 0

    tags = set(git_utils.retrieve_tags(repo))
    for commit in tqdm(data):
        bugtag = 'fixes-{:d}'.format(int(data[commit]['bug_report']['bug_id']))
        if bugtag in tags:
            n_skipped = n_skipped + 1
            continue

        # assumes that 'sha' field is fixed
        git_utils.create_tag(repo, bugtag,
                             data[commit]['commit']['metadata']['sha'])

    # info about process
    print('%d / %d skipped: already tagged' % (n_skipped, len(data)),
          file=sys.stderr)


def data_sort_keys(data):
    """Change data by sorting it's keys according to pre-defined order

    Modifies data in-place.  It can replace dict for nested
    information with OrderedDict.

    The sort order is intended to make it easier to read final JSON
    (where there is an order in which keys are written anyway).

    Parameters
    ----------
        data : dict | OrderedDict
        The combined bug report and repository information from the
        JSON file.
    """
    main_keys_order = {
        'bug_report': 1,
        'commit': 2,
        'views': 3,
    }
    bug_report_order = {
        'id': 1,
        'bug_id': 2,
        'timestamp': 3,
        'summary': 4,
        'description': 5,
        'status': 6,
        'commit': 7,
        'preceding_commit': 8,
        'result': 9,
    }
    commit_order = {
        'metadata': 1,
        'tree_changes': 2,
        'diff': 3,
    }
    commit_metadata_order = {
        'sha': 1,
        'author': 2,
        'date': 4,
        'timestamp': 5,
        'message': 7,
    }
    for commit in tqdm(data):
        # sort 'diff' by pathname, i.e. by keys
        data[commit]['commit']['diff'] = OrderedDict(
            sorted(list(data[commit]['commit']['diff'].items()),
                   key=lambda k, v: k)
        )
        # sort inner keys in specified order
        data[commit]['bug_report'] = OrderedDict(
            sorted(list(data[commit]['bug_report'].items()),
                   key=lambda k, v: bug_report_order.get(k, 999))
        )
        data[commit]['commit']['metadata'] = OrderedDict(
            sorted(list(data[commit]['commit']['metadata'].items()),
                   key=lambda k, v: commit_metadata_order.get(k, 999))
        )
        data[commit]['commit'] = OrderedDict(
            sorted(list(data[commit]['commit'].items()),
                   key=lambda k, v: commit_order.get(k, 999))
        )
        # sort keys in specified order
        data[commit] = OrderedDict(
            sorted(list(data[commit].items()),
                   key=lambda k, v: main_keys_order.get(k, 999))
        )


# helper function from tests/test_git_utils.py
def trim_commit_info(metadata):
    """Trims commit metadata, removing unnecessary fillers

    This function is intended to fix information retrieved from the
    repository by the process_bug_reports.py.  It removes 'commit '
    from the 'sha' field, leaving only full SHA-1 identifier of a
    commit, removes 'Author: ' from the beginning of the author field,
    etc. It also removes unnecesary trailing EOLs.

    Parameters
    ----------
    metadata : dict
        Metadata about commit, as generated by process_bug_reports.py

    Side effects
    ------------
    Modifies its arguments in-place

    Returns
    -------
    dict
        Trimmed commit metadata, in the following form:

        {
            'sha': <commit identifier, as hexadecimal string>,
            'author': <commit author data>,
            'date': <authored date, as text>
            'message': <multiline commit message, without indenting>,
        }
    """
    ## TODO: passthru e.g. via .update(), maybe OrderedDict
    metadata.update({
        'sha': metadata['sha'].replace('commit ','').strip(),
        'author': metadata['author'].replace('Author: ','').strip(),
        'date': metadata['date'].replace('Date: ', '').strip(),
        'message': re.sub(r'\n    ', r'\n',
                          metadata['message'].lstrip()),
    })
    return metadata


if __name__ == '__main__':
    main()
