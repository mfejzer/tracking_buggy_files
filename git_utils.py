# -*- coding: utf-8-unix -*-
"""Utilities to get data out of git repositories.

Uses pygit2 (libgit2 bindings for Python), GitPython and straight-up
calling git commands.

Usage:
------
Example usage:
  >>> from git_utils import retrieve_files
  >>> files = retrieve_files('path/to/git/repo', 'HEAD') # 'HEAD' is the default
  ...     ...

This implementation / backend retrieves data by calling `git` via
`subprocess.Popen`, and parsing the output.

WARNING: at the time this backend does not have error handling implemented;
it would simply return empty result, without any notification about the
error (like incorrect repository path, or incorrect commit)!!!


Example usage with GitPython (using the default pure-Python GitDB backend):
  >>> import git
  >>> import git_utils
  >>> repo = git.Repo('path/to/git/repo')
  >>> files = git_utils.retrieve_files(repo, 'HEAD')
  ... ...

This GitPython backend uses less memory when handling huge files, but
will be 2 to 5 times slower when extracting large quantities small of
objects from densely packed repositories.


Example usage with GitPython (using the GitCmdObjectDB backend):
  >>> import git
  >>> import git_utils
  >>> repo = git.Repo('path/to/git/repo', odbt=git.GitCmdObjectDB)
  >>> files = git_utils.retrieve_files(repo, 'HEAD')
  ... ...

This GitPython backend uses persistent `git-cat-file` instances to
read repository information.  These operate very fast under all
conditions, but will consume additional memory for the process itself.
When extracting large files, memory usage will be much higher than the
one of the `git.GitDB`.


Example usage with pygit2:
  >>> import pygit2
  >>> import git_utils as gt
  >>> repo = pygit2.Repository('path/to/git/repo')
  >>> files = gt.retrieve_files(repo, 'HEAD')

Pygit2 is a set of Python bindings to the libgit2 shared library,
which implements the core of Git.  It requires for libgit2 with the
same version number to be installed.


Tests:
------
Tests for this library can be found in tests/test_git_utils.py.  You
can run them for example with

  $ cd tests
  $ python -m unittest --verbose test_git_utils

Benchmarks:
-----------
Benchmarks of different implementations / backends, either directly
using this library, or using code that this library uses, can be found
in ./timeit.sh script.  This script utilizes Python's `timeit` module.
"""
# git libraries
import pygit2
import git # GitPython

# calling git commands
import subprocess

# parsing
#import re

# datetime
from datetime import datetime
import date_utils


# global variables
DEFAULT_FILE_ENCODING='latin-1'


def _walktree(repo, tree, path=[]):
    ## used for getting list of all files via pygit2 recursive walk
    result = []
    for e in tree:
        if e.type == 'blob':
            result.append('/'.join(path+[e.name]))
        elif e.type == 'tree':
            result.extend(_walktree(repo, repo[e.id], path+[e.name]))
        else:
            # skip submodules
            pass
    return result


def _diff_hunk(hunk):
    ## create hunk start line from pygit2.DiffHunk
    hunk_header = ''.join([
        '@@',
         ' -',str(hunk.old_start),',',str(hunk.old_lines),
         ' +',str(hunk.new_start),',',str(hunk.new_lines),
         ' @@\n'  # no support for so called function header
    ])
    hunk_body = ''.join([
        diffline.origin + diffline.content
        for diffline in hunk.lines
    ])
    return hunk_header + hunk_body


def _diff_with_header(file_diff):
    ## return or re-create diff/patch for a single file from diff data
    # https://git-scm.com/docs/git-diff#_generating_patches_with_p
    # https://stackoverflow.com/questions/2529441/how-to-read-the-output-from-git-diff/2530012#2530012
    global DEFAULT_FILE_ENCODING
    if isinstance(file_diff, git.diff.Diff):
        ## TODO: recreate `diff --git` header
        return file_diff.diff.decode(DEFAULT_FILE_ENCODING)
    elif isinstance(file_diff, pygit2.Patch):
        ## TODO: recreate `diff --git` header
        return ''.join([_diff_hunk(hunk)
                        for hunk in file_diff.hunks])
    else:
        raise NotImplementedError('unsupported file diff / patch type %s (%s)' %
                                  (type(file_diff), file_diff))


def _diff_status(diff):
    ## return one-letter status indicator out of GitPython's diff
    if diff.new_file:
        return 'A'
    elif diff.deleted_file:
        return 'D'
    elif diff.renamed_file:
        # there is no similarity info available, unfortunately
        return 'R'
    else:
        # there is apparently no support for copy detection
        return 'M'


def retrieve_files(repo, commit='HEAD'):
    """Retrieve list of files at given revision in a repository

    Equivalent to running `git ls-tree -r` with appropriate options.
    TODO: ensure that it always return unicode or always str.

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects implementation used.

    commit : str, optional
        The commit for which to list all files.  Defaults to 'HEAD',
        that is the current commit.

    Returns
    -------
    list
        List of full pathnames of all files in the repository.

        Pathnames are :obj:`str` for pygit2 implementation and
        :obj:`unicode` for GitPython and subprocess-based
        implementations.  For GitPython implementation the list
        is not sorted.
    """
    if isinstance(repo, pygit2.repository.Repository):
        return _walktree(repo, repo.revparse_single(commit).tree)
    elif isinstance(repo, git.repo.base.Repo):
        return [e.path for e in
                repo.commit(commit).tree.traverse()
                if e.type == 'blob']
    else:
        process = subprocess.Popen(
            ['git', '-C', repo, 'ls-tree',
             '-r', '--name-only', '--full-tree', '-z',
             commit], stdout=subprocess.PIPE)
        ## NOTE: latin-1 may be wrong for encoding of pathnames
        return process.stdout.read().decode('latin-1').split('\0')[:-1]


def retrieve_changes(repo, commit='HEAD'):
    """Retrieve list of files changed at given revision in repo

    TODO: ensure that it always return unicode or always str.

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

    commit : str, optional
        The commit for which to list changes.  Defaults to 'HEAD',
        that is the current commit.  The changes are relative to
        commit^, that is the previous commit (first parent of the
        given commit).

    Returns
    -------
    list
        List of full pathnames of all changed files in.
        NOTE: those are names in the post-image (new names).

        Pathnames are :obj:`str` for pygit2 implementation and
        :obj:`unicode` for GitPython and subprocess-based
        implementations.

        All implementation do perform rename detection.
    """
    if isinstance(repo, pygit2.repository.Repository):
        diff = repo.diff(commit+'^', commit)
        diff.find_similar()
        return [p.delta.new_file.path for p in diff]
    elif isinstance(repo, git.repo.base.Repo):
        return [d.b_path for d in
                repo.commit(commit+'^').diff(commit)]
    else:
        # --no-commit-id is needed for 1-argument git-diff-tree
        process = subprocess.Popen(
            ['git', '-C', repo, 'diff-tree', '-M',
             '-r', '--name-only', '--no-commit-id', '-z',
             commit], stdout=subprocess.PIPE)
        ## NOTE: latin-1 may be wrong for encoding of pathnames
        return process.stdout.read().decode('latin-1').split('\0')[:-1]


def retrieve_commit(repo, commit='HEAD', ext='.java'):
    """Retrieve selected information about the given commit

    This will extract both metadata about given commit, and
    information about (subset) of differences that the commit
    introduces.

    Returns self-describing nested data structure.

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

    commit : str, optional
        The commit for which to list changes.  Defaults to 'HEAD',
        that is the current commit.  The changes for which the
        information is returned are taken to be relative to commit^,
        that is the previous commit (first parent of the given
        commit).  NOTE: this means that this function would not work
        correctly for the root (parent-less) commits.

    ext : str, optional ('.java' by default)
        Limits information about files changed in given commit to
        files with given extension (with given suffix).  Defaults to
        '.java'.

        Set to `None` to remove path limiting by extension.

    Returns
    -------
    dict
        Nested structure with metadata about commit and with changes.
        Has the following structure:

        {
          'metadata': {
            'sha': 'f8ffd4067d1f1b902ae06c52db4867f57a424f38',
            'author': 'A U Thor <author@example.com>',
            'date': 'Thu Apr 7 16:14:13 2005 -0600',
            'timestamp': 1112912053,
            'message': 'Commit summary\n\nOptional longer description\n',
          },
          'diff': {
            'dir/pathname': '@@ -1,1 +1,1 @@\n-modified\n+MODIFIED\n',
            ...
          }
        }

        The diff is done with no context lines (--unified=0).

        NOTE: does not work correctly for the root commit, as it
        assumes that the commit in question has at least one parent
        commit.
    """
    ## NOTE: may not work correctly for the root commit

    # currently there is no support for subprocess-based solution,
    # because I don't want to re-implement parsing of git-diff output
    # format or commit objects.  Instead we use one of other
    # implementations.
    if isinstance(repo, str):
        repo = git.Repo(repo)
        #repo = pygit2.Repository(repo)

    commit_content = {
        'metadata': retrieve_commit_metadata(repo, commit),
        'diff': retrieve_commit_diff(repo, commit, ext=ext)
    }
    return commit_content


def retrieve_commit_metadata(repo, revision):
    """Retrieve metadata about given commit; part of retrieve_commit()

    Parameters
    ----------
    repo : git.Repo | pygit2.Repository
        Either GitPython (git.Repo) or pygit2 (pygit2.Repository)
        repository object.

        Type of this parameter selects which implementation is used.

    commit : str, optional
        The commit for which to list changes.  Defaults to 'HEAD',
        that is the current commit.

    Returns
    -------
    dict
        Information about selected parts of commit metadata, in
        the following format:

        {
          'sha': 'f8ffd4067d1f1b902ae06c52db4867f57a424f38',
          'author': 'A U Thor <author@example.com>',
          'date': 'Thu Apr 7 16:14:13 2005 -0600',
          'timestamp': 1112912053,
          'message': 'Commit summary\n\nOptional longer description\n',
        }
    """
    if isinstance(repo, git.repo.base.Repo):
        commit = repo.commit(revision)
        a_date = datetime.fromtimestamp(
            commit.authored_date,
            # TODO: create date_utils.GitPythonOffset class
            # commit.author_tz_offset is in seconds west of UTC
            # TODO: check whether git.objects.util.tzoffset or
            # git.objects.util.utctz_to_altz could help here
            date_utils.FixedOffset(-commit.author_tz_offset//60)
        )
        return {
            'sha': commit.hexsha,
            # TODO: committer vs author
            'author': '{} <{}>'.format(commit.author.name,
                                        commit.author.email),
            'timestamp': commit.authored_date,
            # NOTE: %-d (no leading 0) is platform-dependent
            # see
            #   https://stackoverflow.com/questions/28894172/why-does-d-or-e-remove-the-leading-space-or-zero
            #   https://stackoverflow.com/questions/904928/python-strftime-date-without-leading-0
            'date': a_date.strftime('%a %b %-d %H:%M:%S %Y %z'), # Linux, Mac OS X
            #'date': a_date.strftime('%a %b %#d %H:%M:%S %Y %z'), # Windows
            'message': commit.message
        }
    elif isinstance(repo, pygit2.repository.Repository):
        commit = repo.revparse_single(revision)
        a_date = datetime.fromtimestamp(
            commit.author.time,
            # commit.author.offset is offset from UTC in minutes
            date_utils.FixedOffset(commit.author.offset)
        )
        return {
            'sha': commit.hex,
            # TODO: committer vs author
            'author': '{} <{}>'.format(commit.author.name,
                                        commit.author.email),
            'timestamp': commit.author.time,
            # NOTE: %-d (no leading 0) is platform-dependent
            'date': a_date.strftime('%a %b %-d %H:%M:%S %Y %z'), # Linux, Mac OS X
            #'date': a_date.strftime('%a %b %#d %H:%M:%S %Y %z'),  # Windows
            'message': commit.message
        }
    else:
        raise NotImplementedError('unsupported repository type %s (%s)' %
                                  (type(repo), repo))


def retrieve_commit_diff(repo, revision, ext='.java'):
    """Retrieve diff of commit changes; part of retrieve_commit()

    Parameters
    ----------
    repo : git.Repo | pygit2.Repository
        Either GitPython (git.Repo) or pygit2 (pygit2.Repository)
        repository object.

        Type of this parameter selects which implementation is used.

    commit : str, optional
        The commit for which to list changes.  Defaults to 'HEAD',
        that is the current commit.  The changes for which the
        information is returned are taken to be relative to commit^,
        that is the previous commit (first parent of the given
        commit).  NOTE: this means that this function would not work
        correctly for the root (parent-less) commits.

    ext : str, optional ('.java' by default)
        Limits information about files changed in given commit to
        files with given extension (with given suffix).  Defaults to
        '.java'.

        Set to `None` to remove path limiting by extension.

    Returns
    -------
    dict
        Information about changes, where key is path of a file in a
        repository, and the value is diff for this file, without
        extended diff header, and with zero context lines
        (--unified=0).

        For deleted file, the key is path of the file that is deleted;
        the pre-image name.  In all other cases the key is path of the
        file in the post-image.

        Paths are limited to those ending with 'ext', if it is not set
        to `None` (no-op filtering with ext='').
    """
    if isinstance(repo, git.repo.base.Repo):
        commit = repo.commit(revision)
        # diff() method will detect renames automatically
        # we need to turn on patch generation (off by default)
        diff = commit.parents[0].diff(commit, create_patch=True,
                                      paths=['*'+ext] if ext else None,
                                      unified=0, prefix=False)
        return {
            # a path can be false-y if it is None or ''
            file_diff.b_path or
            file_diff.a_path: _diff_with_header(file_diff)
            for file_diff in diff
        }
    elif isinstance(repo, pygit2.repository.Repository):
        # there is no way to turn `--no-prefix` or provide paths
        diff = repo.diff(revision+'^', revision,
                         # we need to turn on renames
                         # ??? - this flag reverses diff ???
                         #flags=pygit2.GIT_DIFF_FIND_RENAMES,
                         context_lines=0)
        diff.find_similar() # find renamed files and update diff in-place
        return {
            # there is pygit2.Diff.patch for a whole diff,
            # but no 'patch' for individual elements, i.e. pygit2.Patch
            file_diff.delta.new_file.path or \
            file_diff.delta.old_file.path: _diff_with_header(file_diff)
            for file_diff in diff
            # there is no support for paths limiting in pygit2's diff()
            if not ext or file_diff.delta.new_file.path.endswith(ext)
        }
    else:
        raise NotImplementedError('unsupported repository type %s (%s)' %
                                  (type(repo), repo))


def retrieve_changes_status(repo, commit='HEAD', prev=None):
    """Retrieve status of file changes at given revision in repo

    It returns in a structured way information equivalent to the one
    from calling 'git diff --file-status -r'.

    Example output:
        {
            (None, 'added_file'): 'A',
            ('file_to_be_deleted', None): 'D',
            ('mode_changed', 'mode_changed'): 'M',
            ('modified', 'modified'): 'M',
            ('to_be_renamed', 'renamed'): 'R'
        }

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

    commit : str, optional
        The commit for which to list changes for.  Defaults to 'HEAD',
        that is the current commit.

    prev : str, optional
        The commit for which to list changes from.  If not set, then
        changes are relative to the parent of 'commit' parameter, which
        means 'commit^'.

    Returns
    -------
    dict
        Information about the status of each change, where the key is
        pair (tuple) of pre-image and post-image pathname, and the value
        is single letter denoting the status / type of the change.

        For new (added) files the pre-image path is None, and for deleted
        files the post-image path is None.

        Possible status letters are:
         - 'A': addition of a file
         - 'C': copy of a file into a new one (not for all implementations)
         - 'D': deletion of a file
         - 'M': modification of the contents or mode of a file
         - 'R': renaming of a file
         - 'T': change in the type of the file (untested)
    """
    if prev is None:
        prev = commit+'^'

    if isinstance(repo, str):
        cmd = [
            'git', '-C', repo, 'diff-tree', '--no-commit-id',
            # turn on renames [with '-M']; note that parsing is a bit
            # easier without '-z', assuming that filenames are sane
            # increase inexact rename detection limit
            '--find-renames', '-l5000', '--name-status', '-r',
            prev, commit
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        ## NOTE: latin-1 may be wrong for encoding of pathnames
        lines = process.stdout.read().decode('latin-1').splitlines()
        result = {}
        for l in lines:
            if l[0] == 'R' or l[0] == 'C':
                status, old, new = l.split("\t")
                result[(old, new)] = status[0] # no similarity info
            else:
                status, path = l.split("\t")
                if status == 'A':
                    result[(None, path)] = status
                elif status == 'D':
                    result[(path, None)] = status
                else:
                    result[(path, path)] = status
        return result

    if isinstance(repo, git.repo.base.Repo):
        # increase inexact rename detection limit with -l5000
        diff = repo.commit(prev).diff(commit, l=5000)
        return {
            # the documentation states that diff.a_path is None for a new file,
            # and diff.b_path is None for deleted file, but this is not true
            # (while it used to work: retrieve_commit_diff() had to handle this
            # case separately; the difference is create_patch=True) ???
            (d.a_path if not d.new_file     else None,
             d.b_path if not d.deleted_file else None): _diff_status(d)
            for d in diff
        }

    elif isinstance(repo, pygit2.repository.Repository):
        diff = repo.diff(prev, commit,
                         # we need to turn on renames
                         # ??? - this flag reverses diff ???
                         #flags=pygit2.GIT_DIFF_FIND_RENAMES,
                         context_lines=0)
        # find renamed files and update diff in-place
        # with increased inexact rename detection limit
        diff.find_similar(rename_limit=5000)
        # it could have been done with list comprehension and
        # conditional operator, but it wouldn't be as readable
        result = {}
        for file_diff in diff:
            delta = file_diff.delta
            if delta.status == pygit2.GIT_DELTA_ADDED:
                src = None
            else:
                src = delta.old_file.path
            if delta.status == pygit2.GIT_DELTA_DELETED:
                dst = None
            else:
                dst = delta.new_file.path
            #if delta.status in (pygit2.GIT_DELTA_RENAMED,\
            #                    pygit2.GIT_DELTA_COPIED):
            #    # note: for some reason it doesn't match other solutions
            #    status = delta.status_char() + \
            #             '{:03d}'.format(delta.similarity)
            #else:
            status = delta.status_char()
            result[(src,dst)] = status
        return result

    else:
        raise NotImplementedError('unsupported repository type %s (%s)' %
                                  (type(repo), repo))


def retrieve_fixes(repo, commit='HEAD'):
    """Retrieve list of files fixed by given bugfix revision in repo

    This is the list of pathnames present in pre-image, i.e. names of files
    before the bugfix.  It excludes files which were added in bugfix commit.

    TODO: ensure that it always return unicode or always str.

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

    commit : str, optional
        The commit for which to list changes.  Defaults to 'HEAD',
        that is the current commit.  The changes are relative to
        commit^, that is the previous commit (first parent of the
        given commit).

    Returns
    -------
    list
        List of full pathnames of all 'fixed' files in commit.
        NOTE: those are names in the pre-image (old names).

        Pathnames are :obj:`str` for pygit2 implementation and
        :obj:`unicode` for GitPython and subprocess-based
        implementations.

        All implementation do perform rename detection.
        TODO: make rename detection configurable
    """
    changes = retrieve_changes_status(repo, commit)
    return [old_path for (old_path, new_path) in list(changes.keys())
            if old_path is not None]


def retrieve_contents(repo, commit, path, encoding=None):
    """Retrieve contents of given file at given revision / tree

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

        NOTE: Both GitPython and pygit2 backends raise KeyError if file
        or commit does not exist; error handling for git command based
        backend is not implemented yet.

    commit : str
        The commit for which to return file contents.  Defaults to
        'HEAD', that is the current commit.

    path : str
        Path to a file, relative to the top-level of the repository

    encoding : str, optional
        Encoding of the file

    Returns:
    --------
    str | unicode
        Contents of the file with given path at given revision
    """
    if encoding is None:
        encoding = DEFAULT_FILE_ENCODING

    if isinstance(repo, str):
        cmd = [
            'git', '-C', repo, 'show',
            #'git', '-C', repo, 'cat-file', 'blob',
            # assumed that 'commit' is sane
            commit+':'+path
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        result  = process.stdout.read().decode(encoding)
        # NOTE: does not handle errors correctly yet
        return result

    elif isinstance(repo, git.repo.base.Repo):
        # first possible implementation, less flexible
        #blob = repo.commit(commit).tree / path
        # second possible implementation, more flexible
        blob = repo.rev_parse(commit + ':' + path)
        result = blob.data_stream.read().decode(encoding)
        return result

    elif isinstance(repo, pygit2.repository.Repository):
        blob = repo.revparse_single(commit + ':' + path)
        result = blob.data
        return result

    else:
        raise NotImplementedError('unsupported repository type %s (%s)' %
                                  (type(repo), repo))


def checkout_revision(repo, commit):
    """Check out given commit in a given repository

    This would usually (and for some cases always) result in 'detached
    HEAD' situation, that is HEAD reference pointing directly to a
    commit, and not to a branch.

    This function is called for its effects and does return nothing.

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

    commit : str
        The commit to check out in given repository.
    """
    ## TODO: implement checking out into separate worktree
    ##
    if isinstance(repo, str):
        cmd = [
            'git', '-C', repo, 'checkout', '-q', commit,
        ]
        # we are interested in effects of the command, not its output
        subprocess.call(cmd)

    elif isinstance(repo, git.repo.base.Repo):
        # GitPython has builtin support only for checking out index and heads
        repo.git.checkout(commit)

    elif isinstance(repo, pygit2.repository.Repository):
        rev = repo.revparse_single(commit)
        # repo.checkout() only accepts refnames / branches
        repo.checkout_tree(rev)
        repo.set_head(rev.id)

    else:
        raise NotImplementedError('unsupported repository type %s (%s)' %
                                  (type(repo), repo))


def create_tag(repo, name, commit='HEAD'):
    """Create lightweight tag (refs/tags/* ref) to the given commit

    NOTE: does not support annotated tags for now; among others it
    would require deciding on tagger identity (at least for some
    backends).

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

    name : str
        Name of tag to be created.  Should follow `git check-ref-format`
        rules for name; for example they cannot contain space ' ',
        tilde '~', caret '^', or colon ':'.

    commit : str, optional
        Revision to be tagged.  Defaults to 'HEAD'.
    """
    if isinstance(repo, str):
        cmd = [
            'git', '-C', repo, 'tag', name, commit,
        ]
        # we are interested in effects of the command, not its output
        subprocess.call(cmd)

    elif isinstance(repo, git.repo.base.Repo):
        repo.create_tag(name, ref=commit)

    elif isinstance(repo, pygit2.repository.Repository):
        if not name.startswith('refs/tags/'):
            name = 'refs/tags/' + name
        repo.references.create(name, repo.revparse_single(commit).id)

    else:
        raise NotImplementedError('unsupported repository type %s (%s)' %
                                  (type(repo), repo))


def retrieve_tags(repo):
    """Retrieve list of all tags in the repository

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

    Returns
    -------
    list
        List of all tags in the repository.
    """
    if isinstance(repo, str):
        process = subprocess.Popen([
            'git', '-C', repo, 'tag', '--list'
        ], stdout=subprocess.PIPE)
        # f.readlines() might be not the best solution
        return [l.rstrip() for l in process.stdout.readlines()]

    elif isinstance(repo, git.repo.base.Repo):
        return [tag.name for tag in repo.tags]

    elif isinstance(repo, pygit2.repository.Repository):
        # there is listall_branches() but no listall_tags()
        # len('refs/tags/') == 10
        return [ref[10:] for ref in repo.listall_references()
                if ref.startswith('refs/tags/')]

    else:
        raise NotImplementedError('unsupported repository type %s (%s)' %
                                  (type(repo), repo))


def find_commit_by_timestamp(repo, timestamp, start_commit='HEAD'):
    """Find first commit in repository older than given date

    Parameters
    ----------
    repo : str | git.Repo | pygit2.Repository
        Pathname to the repository, or either GitPython (git.Repo)
        or pygit2 (pygit2.Repository) repository object.

        Type of this parameter selects which implementation is used.

    timestamp : integer | str
        Date in UNIX epoch format, also known as timestamp format.
        Returned commit would be older than this date.

    start_commit | str, optional
        The commit from which to start walking through commits,
        trying to find the one we want.

    Returns
    -------
    str
        Full SHA-1 identifier of found commit.

        WARNING: there is currently no support for error handling,
        among others for not finding any commit that fullfills
        the condition.  At least it is not tested.
    """
    if isinstance(repo, str):
        cmd = [
            'git', '-C', repo, 'rev-list',
            '--min-age='+str(timestamp), '-1',
            start_commit
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        # this should be US-ASCII hexadecimal identifier
        result  = process.stdout.read().decode('latin-1').strip()
        # NOTE: does not handle errors correctly yet
        return result

    elif isinstance(repo, git.repo.base.Repo):
        # should return at most one element anyway
        for commit in repo.iter_commits(start_commit,
                                        max_count=1, before=timestamp):
            return commit.hexsha

    elif isinstance(repo, pygit2.repository.Repository):
        # return first element matching condition
        for commit in repo.walk(repo.revparse_single(start_commit).id,
                                pygit2.GIT_SORT_TIME):
            # NOTE: for some reason in the version installed there
            # there is .commit_time, but no .author_time, even if docs
            # mention it
            if commit.author.time <= timestamp:
                return commit.id.hex

    else:
        raise NotImplementedError('unsupported repository type %s (%s)' %
                                  (type(repo), repo))


# end of file git_utils.py
