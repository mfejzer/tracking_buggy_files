#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage: %(prog)s [options] <json-file> <repository-path>

Calculates bug fixing recency and bug fixing frequency for all paths
for which is it non-zero from 'bug_report' information in the
<json-file>, with the help of project history from given
<repository-path>.  Prints the result, currently in JSON format, on
its standard output or to given file.  As a side effect it sorts data
by the bug report timestamp (which should be the same as sorting it by
bug id).

The extra information is added to the [possibly new] 'views' object.
This data belongs neither to 'bug_report', not to 'commit', as it can
contain data from one or the other, or even both,

* bug fixing recency,   section 3.4.1:

   f_5(r,s) = (r.month - last(r,s).month + 1)^-1

* bug fixing frequency, section 3.4.2:

   f_6(r,s) = |br(r,s)|

where:
 * r - set of bug reports (for which fix commit exists and is known)
 * s - file path (of a file in a project)
 * br(r,s) - set of bug reports, for which file 's' was fixed before
   report 'r' was created
 * last(r,s) ∈ br(r,s) - most recent previously fixed bug

How is br(r,s) computed, with inductive algorithm
-------------------------------------------------

Let $B$ be a set of bug reports, and $S(t)$ be a set of all files in
the snapshot of the project at given time $t$ (and $S(c)$ set of all
files at given commit $c$).  Let $r \in B$ be a bug report, $r.f$ be a
bugfix commit for a given bug report, and $t_r$ be its creation time.
We will use $changes(c)$ to denote all files changed in commit $c$.

Let's assume that $S(t_r)$ is a good approximation for the version
that bug report was created for (the version that author of the bug
report was using when he/she had found the bug in question). We will
uses $S(r.f)$ for an approximation for this state, as in the original
work.

We can then define $br(r,s)$ in the following way:

\begin{equation}
  \label{eq:br-def}

  br(r,s) = \{b \in B: b.f < t_r \and s(t_b) \in changes(b.f)\)

\end{equation}

It should be noted that the ordering relation $<_{br}$ defined as
$$a <_{br} b   <=>   a.f < t_b$$ does not form strict ordering.
If bug $b$ is created while another bug $a$ is still open, then
neither $a <_{br> b$ not $b <_{br} a$ is true.

The ordering by the bug report date (timestamp) however is is in fact
strict order: $$a <_{rep} b   <=>   t_a < t_b$$.

Let's define $succ(r)$ for $r \in B$ to be the next bug report in the
bug report order after $r$, that is oldest bug report that was created
after $r$.

Let's assume that $fx(a,b)$, where $a, b \in B$ to be a set of bugfix
commits, such that $$c \in fx(a,b)  <=>  t_a <= t(c) <= t_b$$, where
$t(c)$ is the commit timestamp.  NOTE that we will (for now) assume
that if $t(a.f) <= t_b$, for bugfix commit $a.f$ and bug reports $a$
and $b$, then commit $a.f$ could be in $br(b,s)$ for some path $s$.

The inductive algorithm goes as follow:

 * For the first bug report $r_0$ in $B$ (in the order of bug report
   creation date) the set $br(r_0)$ (defined as $br(r,s)$ for all
   paths, that is: $br(r) := {s \in S(r_0): br(r,s)}$) is empty;
   there are no earlier bug reports, thus no bug reports for which
   bug fixing timestamp (which is always later than bug report
   timestamp: $t_a < t(a.f)$ for each $a \in B$) is earlier than
   this bug report.

 * For each subsequent bug report, we update br(r-1) to br(r) in the
   following way

   1. Find all bug fixes that were not taken into account, that is
      candidates = fx(r_last, r)

   2. If not empty, update r_last

   3. For each bug report x in candidates, translate br(*,*)
      to t(x.f^), and then include x it in br(*,t(x.f^))

   4. Translate final result to t(r.f^)

   5. Compute all features dependent on br(r,*)


Here we have used the folloring definition for br(r;t)

\begin{equation}
  \label{eq:br_t-def}

  br(r;t) = \{(b,s) \in S(t) x B:  b.f < r.f \and s(t) \in S(t)
              \and s(b.f^) \in changes(b.f).preimage \}
         ~= \union_{s \in S(t)} br(r,s)
\end{equation}


The description below is taken from "Mapping Bug Reports to Relevant
Files: A Ranking Model, a Fine-Grained Benchmark, and Feature
Evaluation" (chapter numbers before slash) and preceding work
"Learning to Rank Relevant Files for Bug Reports using Domain
Knowledge" (chapter numbers after slash).

3.5 / 3.4.1 Bug-Fixing Recency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The change history of source codes provides information that can help
predict fault-prone files. For example, a source code file that was
fixed very recently is more likely to still contain bugs than a file
that was last fixed long time in the past, or never fixed.

As in Section 3.2, let 'br(r,s)' be the set of bug reports for
which file 's' was fixed before bug report 'r' was created. Let
'last(r,s) ∈ br(r,s)' be the most recent previously fixed bug.
Also, for any bug report 'r', let 'r.month' denote the month
when the bug report was created. We then define the bug-fixing
recency feature to be the inverse of the distance in
months between 'r' and 'last(r,s):

  f_5(r,s) = (r.month - last(r,s).month + 1)^-1      : (8)

Thus, if 's' was last fixed in the same month that 'r' was created,
'f_5(r,s)' is 1. If 's' was last fixed one month before 'r' was
created, 'f_5(r, s)' is 0.5.

3.5 / 3.4.2 Bug-Fixing Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A source file that has been frequently fixed may be a fault-prone
file. Consequently, we define a bug-fixing frequency
feature as the number of times a source file has been fixed
before the current bug report:

  f_6(r,s) = |br(r,s)|                               : (9)

This feature will be automatically normalized during the
feature scaling step described in Section 3.7 below


This view is, as far as I understand it, intended to represent quality
of the code in the given file, based on bug reports and bug fixes.

"""
from __future__ import print_function

import json
import sys
import os
import argparse
from collections import OrderedDict
from datetime import timedelta
from copy import deepcopy

import pandas as pd
from tqdm import tqdm

from git_utils import \
    retrieve_changes, retrieve_files, \
    retrieve_changes_status
from date_utils import convert_commit_date
import args_utils
from datastore_utils import \
    NestedDictStore, select_datastore_backend
from misc_utils import cmp_char

from fix_and_augment import \
    fix_commit_metadata, sorted_by_bugreport, sorted_by_commit
from dataset_utils import \
    list_of_bugfixes_to_storage


br_format = 'list'

def main():
    ### run as script
    global br_format

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Calculate bug report recency and frequency",
        epilog="NOTE: saves stats in the CSV file, if -o/--output provided")
    parser = args_utils.add_arguments_and_usage(parser, repo_required=True)
    parser = args_utils.add_json_output_options(parser)
    parser = args_utils.add_git_backend_selection(parser)
    parser.add_argument('--count-files', action='store_true',
                        help='count files in each bugfix revision')
    parser.add_argument('--br',  default='list',
                        choices=['list', 'sparse', 'bitmap', 'none'],
                        help="storage format for br(r,s)"+
                             " [default '%(default)s']")
    args = parser.parse_args()

    # process arguments
    datafile_json = args.json_file
    repo_path = args.repository_path
    indent_arg  = args.indent
    output_file = args.output or sys.stdout
    br_format = args.br
    # select git backend
    repo = args_utils.repo_by_backend(repo_path, args.git_backend)

    # read and process data, preserving order of keys
    data = args_utils.read_json(datafile_json, preserve_order=True)
    store = select_datastore_backend(args.output.name, category='bug_fixing')
    # augment original bug report data, if supported
    # switch to supported br(r,s) formats
    if isinstance(store, NestedDictStore):
        store.update(data)
        if br_format != 'list' and br_format != 'none':
            print("--br=%s not supported for JSON, switching to 'list'" %
                  (br_format,), file=sys.stderr)
            br_format = 'list'

    store, stats = process_data(store, data, repo, count_files=args.count_files)
    store.print_data(args)
    print('Repository: %s' % (repo_path,),
          file=sys.stderr)
    process_stats(stats, output_file)


def process_stats(stats, output_file):
    """Print summary of statistics and optionally save them to a file

    The data is [optionally] saved to file in CSV (Comma Separated Value)
    format are full per-bug statistics; this is done if `output_file` is
    not sys.stdout.  What is printed is stats summary / description
    of data in the table.

    Parameters
    ----------
    stats : dict
        Statistics for each commit / bug report, to be saved as CSV
        and/or to generate summary of overall statistics of bug
        reports and the repository.  Assumed to be generated by
        `process_data()`.

    output_file : file | sys.stdout
        Original file to which JSON output is saved, or sys.stdout if
        JSON output is to be printed.  If it is not sys.stout, its
        name is used as name of CSV file to saave full statistics to.

    Side effects
    ------------
    Save data to *.stats.csv file if `output_file` is not sys.stdout.
    Print summary of data in `stats` to stderr.
    """
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.sort_values(by='fixes', inplace=True)
    if output_file != sys.stdout:
        stats_filename, __ = os.path.splitext(output_file.name)
        stats_filename += '.stats.csv'
        df.to_csv(stats_filename)
        print("Saved stats to  '%s' (%s)" %
              (stats_filename, args_utils.sizeof_fmt(os.path.getsize(stats_filename))),
              file=sys.stderr)

    df_sum = df.filter(
        items=[
            # commit stats
            'n_changes', 'n_effective',
            'n_renames', 'n_additions', 'n_deletions',
            'n_changes_eff',
            # translation stats
            'n_changes_since',
            'n_deletions_fixed', 'n_renames_fixed',
            # translation to 'r'
            'n_del_fixed_curr', 'n_ren_fixed_curr',
            # bool-valued stats
            'has_prev_in_new',
        ]
    )
    # maybe also  'renames_since', 'deletions_since',\
    for field in ('renames', 'additions', 'deletions',\
                  'renames_fixed', 'deletions_fixed',\
                  'del_fixed_curr', 'ren_fixed_curr'):
        df_sum['with_'+field] = df_sum['n_'+field] > 0
    print(pd.DataFrame(df_sum.sum(),
                       columns=['sum']),
          file=sys.stderr)
    # labels=[...], axis=1 is equivalent to columns=[...], latter from 0.21.0
    #
    # possibly DataFrame.reorder_levels(order=[...], axis=1) instead
    # of using DataFrame.sort_index(axis=1)
    print(df.drop(labels=['fixes', 'timestamp'], axis=1)
            .sort_index(axis=1)
            .describe(percentiles=[.5]).T,
          file=sys.stderr)
    if output_file != sys.stdout:
        stats_filename, __ = os.path.splitext(output_file.name)
        stats_filename += '.description.csv'
        with open(stats_filename, 'w') as stats_file:
            df.drop(labels=['fixes', 'timestamp'], axis=1)\
              .sort_index(axis=1)\
              .describe(percentiles=[.1, .25, .5, .75, .9])\
              .to_csv(stats_file)
            print('',file=stats_file)
            pd.DataFrame(df_sum.sum(),
                         columns=['sum'])\
                         .T\
                         .to_csv(stats_file)

        print("Saved description of stats to  '%s' (%s)" %
              (stats_filename, args_utils.sizeof_fmt(os.path.getsize(stats_filename))),
              file=sys.stderr)


def process_data(store, data, repo, count_files=False):
    """Process bug report dataset, computing bug fixing recency/frequency

    The input to this function is processed bug report dataset, which
    includes all the information about bug reports for fixed bugs.  It
    assumes that data is an associative array (an object), where the
    key is [shortened] SHA-1 identifier of the bugfix commit, and the
    data includes 'bug_report' key.

    Parameters
    ----------
    store : datastore_utils.Store
        Data structure used to store results.

    data : dict
        Data about bug reports, read from the JSON file.

        This structure is modified by this function, getting sorted
        and (depending on 'store') getting data added.

    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

    count_files : bool, optional
        Whether to find number of files in snapshot for each bugfix
        commit (good for statistics).  The number of files is taken at
        pre-image of bugfix commit, that is the state of the project
        at the parent of the bugfix commit (where br(r,s) is evaluated
        at).  This data is saved under 'n_files' key in stats.

    Returns
    -------
    (store, stats) : (datastore_utils.Store, dict)
        A tuple consisting of [augmented] data about bug reports, to be
        printed or written to a file [e.g. as JSON], and statistics for
        each commit / bug report, to be saved as CSV and/or to generate
        summary of overall statistics of bug reports and the repository.
    """
    stats = {}

    # sort data, as the algorithm needs bug_reports sorted by commit
    # date with later fixed bug_reports following earlier ones;
    # see the definition of br(r,s) in the docstring for this file
    print('Fixing and augmenting commit metadata', file=sys.stderr)
    fix_commit_metadata(data)  # ensures 'timestamp' for commit
    print('Sorting data by bug creation timestamp', file=sys.stderr)
    data = sorted_by_bugreport(data, key='timestamp')
    # if augmenting original bug report data, preserve sorting result
    if isinstance(store, NestedDictStore):
        store.update(data)
    # list of bugfix commits sorted by the commit timestamp that were
    # not yet taken into account / used to compute br(r,s)
    br_remaining = sorted(data.keys(),
                          key=lambda c: int(data[c]['commit']['metadata']['timestamp']))
    # HDF5 doesn't have true Unicode support; for PyTables (that's what Pandas
    # uses) on Python 2.x using unicode atrings as names of colums or as index
    # leads to PerformanceWarning about pickling; let's convert to str
    # (i.e. bytes).  NOTE: this may change for Python 3.x
    #
    # NOTE: SHA-1 hash in hex is always in 'ascii' range and encoding
    br_remaining = map(str, br_remaining)
    br_new = []

    print('Computing br(r,t)', file=sys.stderr)
    prev_commit = None
    br_data = {}  # translated to br_last
    br_curr = {}  # translated to current commit
    br_last = None

    # DEBUG/debugging (also possibly for 'bitmap' br(r,s) storage)
    commit_idx = { commit: idx for (idx,commit) in enumerate(data) }

    # debugging strange patterns in the dataset (printing needs also `commit_idx`)
    n_suspicious = 0            # number of times br_commit was not in stats
    n_bug_time_collisions = 0   # encountered bug reports with the same date (maybe dup)
    n_fix_before_report = 0     # bug reports where the fix is earlier than its creation
    # TODO: make max_warnings configurable and/or use logging
    max_warnings = 40           # max number to be warned, < 0 for unlimited, 0 to supress

    for commit in tqdm(data):
    # DEBUG
    #for (idx, commit) in enumerate(tqdm(data)):
        # update/find stats about bugfix commit
        commit_full = data[commit]['commit']['metadata']['sha']
        fixstatus = retrieve_changes_status(repo, commit_full)
        init_bugfix_changes_stats(stats, data, commit,
                                  changes=fixstatus)
        create_commit_info_stats(stats, data, commit)

        # debugging strange patterns in the dataset
        if int(data[commit]['commit']['metadata']['timestamp']) < \
           int(data[commit]['bug_report']['timestamp']):
            n_fix_before_report += 1

        # ensure that there is 'len_br_preceding', even for first bugfix
        # which means that there is always 'len_br_preceding' for
        # previous bugfix commit (needed to calculate one for current)
        stats[commit]['len_br_preceding'] = 0

        # if there were no previous bug reports, then br(r,t(r)) is empty
        if prev_commit is None:
            prev_commit = commit
            continue

        # find all new commits that may be included in br(r,t(r)), that
        # is those that were not yet included in BR(r), and which t(b.f) <= t_r,
        # i.e. those that were fixed before current bug report was created
        if len(br_new) > 0:
            br_last = br_new[-1]
        (br_new, br_remaining) = br_candidates(data, br_remaining, commit)

        # DEBUG
        # TODO: if __debug__: or logger instead of commenting out code
        #print('%d remaining + %d candidates = %d vs %d' %
        #      (len(br_remaining), len(br_new), len(br_remaining)+len(br_new), len(data)))
        #print('br_last=%s commit=%s (%d); %d candidates=%s' %
        #      (br_last, commit, idx, len(br_new), br_new))

        # TODO: move to separate function
        stats[commit]['len_br_new'] = len(br_new)
        stats[commit]['len_br_preceding'] = \
            stats[prev_commit]['len_br_preceding'] + stats[commit]['len_br_new']
        # TODO?: this may be done by comparing timestamps instead
        stats[commit]['has_prev_in_new'] = \
            len(br_new) > 0  and  prev_commit in br_new

        # we might need to translate br to current commit
        if not br_new and not br_data:
            # no new bugfixes, no br, nothing to translate
            continue

        # state for bug report is assumed to be parent of bugfix commit
        # NOTE: prev_commit cannot be None, otherwise we wouldn't get here
        prev_br_commit = br_last
        for br_commit in br_new:
            # translate to next commit, if there us anything to translate
            # and if there was a previous commit to translate from
            if br_data and prev_br_commit is not None:
                prev_br_commit_full = data[prev_br_commit]['commit']['metadata']['sha']
                br_commit_full = data[br_commit]['commit']['metadata']['sha']
                # files might have been deleted, added or renamed
                diffstatus = retrieve_changes_status(repo,
                                                     commit=br_commit_full+'^',
                                                     prev=prev_br_commit_full+'^')
                changes_stats = \
                    br_translate(br_data, changes=diffstatus)
                # update bugfix commit stats
                if br_commit not in stats:
                    # debugging strange patterns in the dataset
                    n_suspicious += 1
                    if int(data[commit]['bug_report']['timestamp']) == \
                       int(data[br_commit]['bug_report']['timestamp']):
                        n_bug_time_collisions += 1

                    # kind of DEBUG, supressable
                    if max_warnings > 0 and n_suspicious < max_warnings:
                        print('%s* commit=%s [%d/id=%d] timestamps: bug=%d %s commit=%d' %
                              (cmp_char(int(data[commit]['bug_report']['timestamp']),
                                        int(data[br_commit]['bug_report']['timestamp'])),
                               commit, commit_idx[commit],
                               int(data[commit]['bug_report']['id']),
                               int(data[commit]['bug_report']['timestamp']),
                               cmp_char(int(data[commit]['bug_report']['timestamp']),
                                        int(data[commit]['commit']['metadata']['timestamp'])),
                               int(data[commit]['commit']['metadata']['timestamp'])),
                              file=sys.stderr)
                        print('br_commit=%s [%d/id=%d] timestamps: bug=%d %s commit=%d' %
                              (br_commit, commit_idx[br_commit],
                               int(data[br_commit]['bug_report']['id']),
                               int(data[br_commit]['bug_report']['timestamp']),
                               cmp_char(int(data[br_commit]['bug_report']['timestamp']),
                                        int(data[br_commit]['commit']['metadata']['timestamp'])),
                               int(data[br_commit]['commit']['metadata']['timestamp'])),
                              file=sys.stderr)

                    stats[br_commit] = {}
                # actual update
                stats[br_commit].update(changes_stats)

            # add commit to br
            br_commit_full = data[br_commit]['commit']['metadata']['sha']
            fixstatus = retrieve_changes_status(repo, br_commit_full)
            changes_effective = \
                br_add_bugfix(br_data, bugfix=br_commit, changes=fixstatus)
            stats[br_commit]['n_changes_eff'] = len(changes_effective)

            # update previous
            prev_br_commit = br_commit

        # translation to current commit
        br_curr = deepcopy(br_data)
        br_commit_full = data[br_commit]['commit']['metadata']['sha']
        diffstatus = retrieve_changes_status(repo,
                                             commit=commit_full+'^',
                                             prev=br_commit_full+'^')
        changes_stats = br_translate(br_curr, changes=diffstatus)
        stats[commit].update({
            'n_ren_to_curr': changes_stats['n_renames_since'],
            'n_del_to_curr': changes_stats['n_deletions_since'],
            'n_ren_fixed_curr': changes_stats['n_renames_fixed'],
            'n_del_fixed_curr': changes_stats['n_deletions_fixed'],
        })

        # compute and store dependent features
        stats_br = \
            br_compute_frequency_and_recency(store, br_curr, data, commit)
        stats[commit].update(stats_br)

        prev_commit = commit

    # debugging strange patterns in the dataset
    n_total = len(data)
    if n_suspicious > 0:
        print('%d/%d suspicious bug reports, with br_commit not in stats' %
              (n_suspicious, n_total), file=sys.stderr)
        if max_warnings > 0:
            print('- supressed warnings after %d' % max_warnings,
                  file=sys.stderr)
    if n_bug_time_collisions > 0:
        print('%d/%d occurences where two bug reports had the same timestamp' %
              (n_bug_time_collisions, n_total), file=sys.stderr)
    if n_fix_before_report > 0:
        print('%d/%d bug reports was fixed before bug report creation date' %
              (n_fix_before_report, n_total), file=sys.stderr)

    if count_files:
        print('Finding number of files in each bugfix commit', file=sys.stderr)
        for commit in tqdm(data):
            commit_full = data[commit]['commit']['metadata']['sha']
            files = retrieve_files(repo, commit_full+'^')
            # TODO?: add this information to data, and save it
            stats[commit]['n_files'] = len(files)
            stats[commit]['n_files_Java'] = len([f for f in files if f.endswith('.java')])
            # TODO: compute mean and/or median length of pathname

    else:
        prev_commit_full = data[prev_commit]['commit']['metadata']['sha']
        print('number of files in repository at last bugfix %s: %d' %
              (prev_commit, len(retrieve_files(repo, prev_commit_full))),
              file=sys.stderr)

    # NOTE: currently data is not getting sorted
    return (store, stats)


def br_add_bugfix(fixing_data, bugfix, changes):
    """Update bug fixing recency data with the new bugfix

    This function simply adds `bugfix` to the list of bugfixes/commits
    for each file that the bugfix modified, that is each file in the
    pre-image of `changes`, creating entries if necessary.

    Used to extend 'br(r-1;t(r.f^))' with 'r' to 'br(r;t(r.f^))'.

    Parameters
    ----------
    fixing_data : dict
        Per-file bug fixing info; it should have the following
        structure:

            {
                '<pathname>': [
                    '<bugfix_commit_1 shortened sha-1>',
                    '<bugfix_commit_2 shortened sha-1>',
                    ...
                ],
                ...
            }

    bugfix : str
        Identifier of the bugfix commit we are (possibly) appending to
        'br'; a key to data structure with combined bug report and bug
        fixing info.

    changes : dict
        Result of calling retrieve_changes_status() for bugfix commit;
        key is pair of pre-commit and post-commit pathnames, while
        the value is one-letter status of changes.

            {
                (None, '<added file pathname>'): 'A',
                ('<pathname of file to be deleted>', None): 'D',
                ('<pathname of modified file>', '<pathname of modified file>'): 'M',
                ('<pre-change pathname of renamed file>', '<post-change pathname>'): 'R'
            }

    Returns
    -------
    list
        Effective changes, that is list of pathnames for which given
        bugfix changed list of bugfixing (changed fixing information).

    Side effects
    ------------
    Changes / updates 'fixing_data' parameter

    """
    changes_effective = []
    for (src, dst), status in changes.items():
        path = src
        if status == 'A':
            # create info for newly added files
            path = dst
        elif status == 'D':
            # skip deleted files, those would be handled later
            # or in other words deleting file means fixing it
            continue

        # it may be new file, or first time file was fixed
        if path not in fixing_data:
            # NOTE: we can use dict instead of OrderedDict here
            fixing_data[path] = []

        # update information
        changes_effective.append(path)
        fixing_data[path].append(
            # information about bug report / bug fixing commit
            # this is the very simplest solution: just identifier
            bugfix
        )

    return changes_effective


def br_translate(fixing_data, changes):
    """Translate bug fixing recency data from one snapshot to the other

    This means that entries for files that got deleted during
    the transition are removed from bug fixing recency data, and
    entries for files that got renamed have their pathname (their key)
    respectively changed.

    In other words, translate 'br(r,before)' to 'br(r,after)',
    where changes = diff(before, after)

    Parameters
    ----------
    fixing_data : dict
        Per-file bug fixing info; it should have the following
        structure:

            {
                '<pathname>': [
                    '<bugfix_commit_1 shortened sha-1>',
                    '<bugfix_commit_2 shortened sha-1>',
                    ...
                ],
                ...
            }

    changes : dict
        Changes from 'before' commit/snapshot to 'after' commit/snapshot.
        Result of retrieve_changes_status(repo, before, after);
        key is pair of pre-commit and post-commit pathnames, while
        the value is one-letter status of changes.

    Returns
    -------
    dict
        Statistics summary for the change
    """
    stats = {
        'n_renames_since':   0,
        'n_deletions_since': 0,
        'n_renames_fixed':   0,
        'n_deletions_fixed': 0,
    }
    #
    # remove information about files which got deleted
    for path in [srcdst[0]  # the src pathname
                 for srcdst, status in changes.items()
                 if status == 'D']:
        stats['n_deletions_since'] += 1
        # the deleted file might not have been in fix commit
        if path in fixing_data:
            del fixing_data[path]
            stats['n_deletions_fixed'] += 1

    # update information about renamed and copied files
    for src, dst in [srcdst  # (src_path, dst_path) pair
                     for srcdst, status in changes.items()
                     if status in 'RC']:
        stats['n_renames_since'] += 1
        # renamed file might not have been in bugfix commit
        if src not in fixing_data:
            continue
        if changes[(src,dst)] == 'R':
            # move data to the new name, i.e. change the key
            # NOTE: this trick works for regular dict only
            fixing_data[dst] = fixing_data.pop(src)
            stats['n_renames_fixed'] += 1
        else:  # it is 'C'
            # copy data to new name
            fixing_data[dst] = deepcopy(fixing_data[src])

    # information about new files added only as necessary

    return stats


def br_candidates(data, br_remaining, commit):
    """Find candidates not yet included to be added to br(r,*)

    Given the list of remaining, that is not yet taken into account
    bug fixes, split this list into part that has been created (fixed)
    before creation time of given bug report, and those that were
    created (fixed) later.

    Creation times of bugfix commits, and the date when given bug
    report was creates is taken from the augmented combined bugs+fixes
    dataset.  The list of remaining fixes (as shortened SHA-1
    identifiers, which are keys in the dataset to bug report + bug fix
    info) needs to be sorted in ascending chronological order of
    bugfix commit creation date.  Returned lists are also sorted; the
    original list is split in two.

    Parameters
    ----------
    data : dict | collections.OrderedDict
        Combined data about bug reports and bugfix commits, read from
        the JSON file.

    br_remaining : list
        List of remaining keys to data (of shortened SHA-1
        ideintifiers of bugfix commits), sorted in the bugfix commit
        creation time order.  This means that the `commit` timestamp
        divides this list into two parts: first that has commit
        creation date not later than creation date of given bugfix,
        and those that are later.

                             /-- t(r)
                             |
          [c_0, c_1,...,c_i, | c_{i+1},...,c_{N-1}]

        where t(c_j) < t_(c_{j+1}) and t(c_i) < t(r) <= t(c_{i+1}).

    commit : str
        Identifier of the bug report, all bugfix commits added to the
        returned list have commit date not later than bug report
        creation date.

        TODO?: maybe change the name of this parameter.

    Returns
    -------
    (br_new, br_remaining) : (list, list)
        First list in returned pair is the list of bugfix commits from
        `br_remaining` with creation time earlier than bug report
        creation time of `commit`.  Because `br_remaining` is assumed
        to be sorted this would be some number of elements from the
        start of it, and it would also be sorted.  Possibly empty.

        Second list in returned pair is the list of remaining bugfix
        commits, with creation time later than cration time of given
        bug report.  Possibly empty.

        These two lists are (br_remaining[:i], br_remaining[i:]) for
        some value of i.

    """
    this_bug_ts = int(data[commit]['bug_report']['timestamp'])
    this_fix_ts = int(data[commit]['commit']['metadata']['timestamp'])
    commit_list = []

    # DEBUG
    #print('commit  =%s (bug_ts=%d) / bug_id=%d' %
    #      (commit, this_bug_ts, int(data[commit]['bug_report']['bug_id'])))

    # corner cases
    if not br_remaining:
        # DEBUG
        #print('br_candidates: empty list')
        # no candidates
        return ([], [])
    elif this_bug_ts <= int(data[br_remaining[0]]['commit']['metadata']['timestamp']):
        # DEBUG
        #print('br_candidates: early return %d < %d' %
        #      (this_bug_ts, int(data[br_remaining[0]]['commit']['metadata']['timestamp'])))
        # all commits are later (newer) than given bug
        return ([], br_remaining)
    elif int(data[br_remaining[-1]]['commit']['metadata']['timestamp']) < this_bug_ts:
        # even last commit is earlier (older) than given bug
        # NOTE: should never happen in this code
        return (br_remaining, [])

    for (i,v) in enumerate(br_remaining):
        curr_bug_ts = int(data[v]['bug_report']['timestamp'])
        curr_fix_ts = int(data[v]['commit']['metadata']['timestamp'])

        if not curr_fix_ts < this_bug_ts:
            return (br_remaining[:i], br_remaining[i:])


def br_compute_frequency_and_recency(store, bug_fixing_data, data, commit):
    """Compute bug-fixing frequency and bug-fixing recency features

    Compute fixing recency:

        f_5(r,s) = (r.month - last(r,s).month + 1)^-1

    and bug fixing recency:

        f_6(r,s) = |br(r,s)|

    given br(r,s) data for some bug report / bugfix commit.  Bug
    report creation date, and the time it was fixed (bugfix commit
    creation date) is taken from the dataset.

    These computed features, and also some data used to calculate
    them, is saved in given data store.

    Return statistics about those two features.

    TODO: fix order of parameters, rename some of them.

    Parameters
    ----------
    store : datastore_utils.Store
        Data structure used to store results.

    bug_fixing_data : dict
        Per-file bug fixing info, used to compute derived features

    data : dict
        Data about bug reports, read from the JSON file.

    commit : str
        Identifier of the bugfix commit, relevant for computing recency
        value, and where to store results

    Side effects
    ------------
    Appends data to store

    Returns
    -------
    dict
        Statistics about bug frequency, bug recency and related stuff.
    """
    global br_format

    stats = {
        'files_with_br': 0,
        'max_frequency': 0,
    }
    this_bug_ts = int(data[commit]['bug_report']['timestamp'])
    this_fix_ts = int(data[commit]['commit']['metadata']['timestamp'])
    for path in bug_fixing_data:
        frequency = len(bug_fixing_data[path])
        if frequency == 0:
            # should not happen
            continue

        # TODO: better selection, e.g. dict of functions
        if br_format == 'list':
            store.set_feature(commit, path, 'br', bug_fixing_data[path])
        elif br_format == 'bitmap':
            store.set_feature(commit, path, 'br',
                              list_of_bugfixes_to_storage(data,
                                                          bug_fixing_data[path]))
        elif br_format != 'none':
            store.set_br(data, commit, path, 'br', bug_fixing_data[path])
        store.set_feature(commit, path, 'frequency', frequency)

        stats['files_with_br'] += 1
        stats['max_frequency'] = max(stats['max_frequency'],
                                     frequency)

        lastbr = bug_fixing_data[path][-1]  # TODO: not exactly
        last_bug_ts = int(data[lastbr]['bug_report']['timestamp'])
        last_fix_ts = int(data[lastbr]['commit']['metadata']['timestamp'])
        delta_ts = this_bug_ts - last_fix_ts

        # DEBUG (on error)
        #if delta_ts < 0:
        #    print("path='%s'" % (path,))
        #    print('br.last=%s (tc=%d) curr=%s (tr=%d) delta[s]=%d' %
        #          (lastbr, last_fix_ts, commit, this_bug_ts, delta_ts))
        #    sys.exit(0)

        store.set_feature(commit, path, 'recency_timedelta[s]', delta_ts)
        store.set_feature(commit, path, 'recency[30-day months]',
                          1.0/((timedelta(seconds=delta_ts).days//30) + 1))
        # TODO: use relativedelta or rrule from dateutil
        # see https://stackoverflow.com/questions/4039879/best-way-to-find-the-months-between-two-dates

    return stats


def init_bugfix_changes_stats(stats, data, commit, changes):
    """Initialize stats related to changes in the bugfix commit

    Those statistics include currently:
     - n_changes:   total number of all changed files
     - n_effective: all changes except deletions
     - n_renames:   all file renames
     - n_deletions: all file deletions
     - n_additions: all file additions (new files in the commit)

    Parameters
    ----------
    stats : dict
        Statistics for each commit / bug report, to be saved as CSV
        and/or to generate summary of overall statistics of bug
        reports and the repository.

    data : dict
        Combined data about bug reports and bugfix commits, read from
        the JSON file.

    commit : str
        Identifier of the bugfix commit, key to 'stats' structure
        to create and/or update.

    changes : dict
        Result of calling retrieve_changes_status() for a commit;
        key is pair of pre-commit and post-commit pathnames, while
        the value is one-letter status of changes.

    Returns
    -------
    dict
        Updated statistics data

    Side effects
    ------------
    Modifies contents of 'stats' parameter
    """
    if commit not in stats:
        stats[commit] = {}
    stats[commit].update({
        'n_changes': len(changes),
        'n_effective': len([pair for pair, status in changes.items()
                            if status != 'D']),
        'n_renames':   len([pair for pair, status in changes.items()
                            if status == 'R']),
        'n_deletions': len([pair for pair, status in changes.items()
                            if status == 'D']),
        'n_additions': len([pair for pair, status in changes.items()
                            if status == 'A']),
    })

    return stats


def create_commit_info_stats(stats, data, commit):
    """Create statistics info related to bug and bugfix commit itself

    Those statistics and info include currently:
     - fixes:       bug report id of bug fixed by given commit
     - timestamp:   bugfix commit creation time

    Parameters
    ----------
    stats :
        Statistics for each commit / bug report, to be saved as CSV
        and/or to generate summary of overall statistics of bug
        reports and the repository.

        This parameter is modified by this function.

    data :
        Combined data about bug reports and bugfix commits, read from
        the JSON file.

    commit :
        Identifier of the bugfix commit, key to 'stats' structure
        to create and/or update.

    Returns
    -------
    dict
        Modified and augmented stats data.
    """
    if commit not in stats:
        stats[commit] = {}
    stats[commit].update({
        'fixes': int(data[commit]['bug_report']['bug_id']),
        'timestamp': float(data[commit]['commit']['metadata']['timestamp']),
    })
    return stats


def init_changes_since_stats(stats, commit, changes):
    """Initialize stats related to changes since previous bugfix

    Parameters
    ----------
    stats :
        Statistics for each commit / bug report, to be saved as CSV
        and/or to generate summary of overall statistics of bug
        reports and the repository.

        This parameter is modified by this function.

    commit :
        Identifier of the bugfix commit, key to 'stats' structure
        to create and/or update.

    changes :
        Result of calling retrieve_changes_status() for a transition
        (translation) between two commits, assumed to be current and
        previous bugfix.  The key in this structure is pair of
        pre-commit and post-commit pathnames, while the value is
        one-letter status of changes.

    Returns
    -------
    dict
        Modified and augmented stats data.
    """
    if commit not in stats:
        stats[commit] = {}
    stats[commit].update({
        'n_changes_since': len(changes),
    })
    return stats


if __name__ == '__main__':
    main()
