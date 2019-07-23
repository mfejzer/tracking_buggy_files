# -*- coding: utf-8-unix -*-
"""Utilities to help handle program arguments

This module is not intended to be something generic; it makes it
easier to create common options used in this project, for example for
reading from JSON, and for saving data to JSON file or printung it as
JSON.

It also has functions to read data from JSON, and to save data as
JSON.
"""
from __future__ import print_function

# parsing arguments
import argparse
import os
import sys

# reading and writing data
import json
from collections import OrderedDict

# datetime
from datetime import datetime
import date_utils

# running git, for choosing backend
import git
import pygit2

# main script
import __main__


class usage_and_exit(argparse.Action):
    """Class for creating `--usage` option printing file docstring

    This class helps to create something similar to the automatically
    generated -h/--help option, but instead of composing help text out
    of program description, options and their help text, and epilog,
    it shows docstring of the main script.  It is responsibility of
    the programmer to put something sensible there.

    The main file docstring can include various format specifiers to
    avoid repetition/specifying of things like the program name.
    Currently the available specifiers include the program name,
    %(prog)s.

    Usage:
    ------
    >>> parser = argparse.ArgumentParser()
    ... ...
    >>> parser.add_argument('--usage', action=usage_and_exit,
    >>>                     help='show usage and full description, and exit')
    >>> args = parser.parse_args()

    """
    ### based on code of argparse._HelpAction
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(usage_and_exit, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        ### print script usage and exit program
        # %(prog)s is also argparse specifier for the program name
        print(__main__.__doc__ % {'prog': os.path.basename(sys.argv[0])})
        parser.exit()


def add_arguments_and_usage(parser, repo_required=False):
    """Add <json-file>, <repository-path> and --usage to arguments

    The <repository-path> positional argument might be missing, might
    be optional, and might be made mandatory.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser that would define how command-line
        arguments should be parsed.  Needs to support add_argument()
        method.

    repo_required : bool | None, optional
        Decides how <repository-path> positional argument should be
        treated.  If it is None, then <repository-path> argument would
        not be added.  Otherwise, if it is false (the default), then
        the argument in question would be made optional, if true then
        it would be made required.

    Returns
    -------
    argparse.ArgumentParser
        The original ArgumentParser with appropriate arguments added.
        To be further used as e.g. args = parser.parse_args()
    """
    parser.add_argument('json_file', metavar='<json-file>',
                        help='data file in JSON format to process')
    if repo_required is None:
        pass
    elif repo_required:
        parser.add_argument('repository_path', metavar='<repository-path>',
                            help='path to the project repository')
    else:
        parser.add_argument('repository_path', metavar='<repository-path>',
                            nargs='?',
                            help='optional path to the project repository')
    parser.add_argument('--usage', action=usage_and_exit,
                        help='show usage and full description, and exit')
    return parser


def add_json_output_options(parser):
    """Add --output=<filename> and JSON formatting options

    It is intended to be used together with print_json() function;
    see the "Examples" section below.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser that would define how command-line
        arguments of the program should be parsed.  Needs to support
        the add_argument() method.

    Returns
    -------
    argparse.ArgumentParser
        The original ArgumentParser with appropriate arguments added.
        To be further used as e.g. args = parser.parse_args()

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> parser = args_utils.add_json_output_options(parser)
    >>> args = parser.parse_args()
    >>> args_utils.print_json({'key': 'value', 'array': [1, 2]}, args)
    Saving/printing data set with 2 elements
    {"array": [1, 2], "key": "value"}

    >>> parser.parse_args(['-h'])
    usage: ipython [-h] [-o OUTPUT] [-i [INDENT] | -n]

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTPUT, --output OUTPUT
                            The output file to which write augmented data. Should
                            be different from input files, as it would be cleared
                            on start of the program. If not specified, writes to
                            stdout.

    formatting JSON output:
      -i [INDENT], --indent [INDENT]
                            Number of spaces to indent JSON [default: 4]
      -n, --no-indent       Produce JSON as single line (the default)
    """
    # output file
    parser.add_argument('-o', '--output', default=sys.stdout,
                        type=argparse.FileType('w'),
                        help='The output file to which write augmented data. '+
                             'Should be different from input files, '+
                             'as it would be cleared on start of the program. '+
                             'If not specified, writes to stdout.')
    # JSON formatting
    pretty = parser.add_argument_group('formatting JSON output')
    group = pretty.add_mutually_exclusive_group()
    group.add_argument('-i', '--indent', type=int, nargs='?', const=4,
                       help='Number of spaces to indent JSON [default: 4]')
    group.add_argument('-n', '--no-indent', dest='indent',
                       action='store_const', const=None, default=None,
                       help='Produce JSON as single line (the default)')
    return parser


def add_git_backend_selection(parser):
    """Add --backend=('cmd'|'gitpython'|'pygit2') option to parser

    This option is intended to make it easy to choose between
    implementation based on calling git commands ('cmd'), one using
    GitPython (with pure-Python GitDB implementation), and one using
    pygit2 wrapper to libgit2 library.

    The result of parsing arguments with this option is intended to be
    passed to repo_by_backend() function.

    Parameters
    ----------
        parser : argparse.ArgumentParser
        The ArgumentParser that would define how command-line
        arguments of the program should be parsed.  Needs to support
        the add_argument() method.

    Returns
    -------
    argparse.ArgumentParser
        The original ArgumentParser with appropriate arguments added.
        To be further used as e.g. args = parser.parse_args()

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> parser = args_utils.add_git_backend_selection(parser)
    >>> args = parser.parse_args()
    ... ...
    >>> repo = args_utils.repo_by_backend(repo_path, args.git_backend)
    ... ...
    >>> files = git_utils.retrieve_files(repo)
    """
    parser.add_argument('--backend', default='cmd', dest='git_backend',
                        choices=['cmd','gitpyton','pygit2'],
                        help="backend for getting data out of git repositories"+
                             " [default '%(default)s']")
    ## TODO: add custom action to actually select backend,
    ## without requiring to call a separate function, if possible
    return parser


def read_json(filename, preserve_order=False):
    """Reads data from the JSON file, optionally preserving order

    Parameters:
    -----------
    filename : str | '-' | file
        Pathname of the JSON file to read, or opened file; it is
        passed to :obj:`safeopen` for opening, which means that '-'
        denotes reading from standard input.

    preserve_order : bool
        Whether to preserve order of elements in JSON file.  Without it
        the function returns `dict`, with it `collections.OrderedDict`.

    Returns:
    --------
    dict | OrderedDict | list
        The data read from the JSON file.  In the case of this project
        and the JSON file structure it would be dict or OrderedDict.

    Side effects:
    -------------
    Prints information about the progress to the stderr.
    """
    with safeopen(filename) as datafile:
        # progress info
        if filename != '-':
            print("Loading %s data file: '%s'" %
                  (sizeof_fmt(os.path.getsize(filename)), filename),
                  file=sys.stderr)
        else:
            print("Reading data from standard input",
                  file=sys.stderr)
        # read file
        if preserve_order:
            data = json.load(datafile,
                             object_pairs_hook=OrderedDict)
        else:
            data = json.load(datafile)

    return data


def print_json(data, args):
    """Print data as JSON file according to command-line arguments

    It is intended to be used together with `add_json_output_options()`.
    Prints data in JSON format to file or on stdout.

    Parameters
    ----------
    data : dict | list | ...
        Data, usually a nested structure, to be printed or saved in
        JSON format

    args : argparse.Namespace
        Result of running parser.parse_args(), that is expected to
        include options about formatting JSON, results of running
        add_json_output_options(parser).  This includes args.indent
        and args.output

    Side effects
    ------------
    Prints information about progress to standard output

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> parser = args_utils.add_json_output_options(parser)
    >>> args = parser.parse_args()
    ... ...
    >>> args_utils.print_json(data, args)
    """
    indent_arg  = args.indent
    output_file = args.output or sys.stdout

    # progress info
    # note that sys.getsizeof(data) is not recursive
    print(#"Saving/printing %s data set with %d elements" %
          #(sizeof_fmt(sys.getsizeof(data)), len(data)),
          "Saving/printing data set with %d elements" %
          (len(data)),
          file=sys.stderr)
    # write output/file
    print(json.dumps(data, indent=indent_arg,
                     separators=(',', ': ') # if indent_arg is set
                     if indent_arg is not None else None),
          file=output_file)
    # progress info
    if output_file != sys.stdout:
        output_file.close()
        print("Saved output to '%s' (%s)" %
              (output_file.name, sizeof_fmt(os.path.getsize(output_file.name))),
              file=sys.stderr)


def repo_by_backend(repo_path, backend_name='cmd'):
    """Return repository 'object' for given backend

    Parameters
    ----------
    repo_path : str
        Path to git repository (many backends require for it to be
        exact path; there woul be no search)

    backend_name : 'cmd' | 'gitpython' | 'pygit2', optional
        Name of backend to use
         - 'cmd' means running git commands with subprocess module and
           parsing their output
         - 'gitpython' uses https://github.com/gitpython-developers/GitPython
         - 'pygit2' uses https://github.com/libgit2/pygit2
           (Python bindings to libgit2 library)

    Returns
    -------
    str | git.Repo | pygit2.Repository
        Repository object for 'gitpython' and 'pygit2' backend,
        pathname to the repository for 'cmd' backend.

        This result is suitable as `repo` parameter to functions from
        the git_utils module.
    """
    if backend_name == 'gitpython':
        return git.Repo(repo_path)
    elif backend_name == 'pygit2':
        return pygit2.Repository(repo_path)

    # fallback
    return repo_path


## ......................................................................
## helper functions
def safeopen(name, mode='r', buffering=1):
    """Returns open file, given file or filename, special-casing '-'

    Using '-' as filename means standard input for reading, standard
    output for writing.

    Parameters
    ----------
    name : str | file
        If this argument is string, it is treated as a pathname of the
        file to open.  The special name '-' means stdout when file is
        to be used for writing, stdin when file is to be used for
        reading.

        If this argument is :obj:`file`, it is assumed to be already
        opened in appropriate mode.

    mode : str
        The 'mode' passed to `open(name[, mode[, buffering]]) call.

        The mode can be 'r', 'w' or 'a' for reading (default), writing
        or appending.  The file will be created if it doesn't exist
        when opened for writing or appending; it will be truncated
        when opened for writing.  Add a 'b' to the mode for binary
        files.  Add a '+' to the mode to allow simultaneous reading
        and writing.  Add a 'U' to mode to open the file for input
        with universal newline support.  'U' cannot be combined with
        'w' or '+' mode.

    buffering : int, optional
        The 'buffering' passed to `open(name[, mode[, buffering]]) call.

        If the buffering argument is given, 0 means unbuffered, 1
        means line buffered (which is the default), and larger numbers
        specify the buffer size.
    """
    if isinstance(name, file):
        return name
    elif name == '-':
        if mode == 'r':
            return sys.stdin
        else:
            return sys.stdout
    else:
        return open(name, mode, buffering)


# https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
# https://stackoverflow.com/a/1094933/46058
def sizeof_fmt(num, suffix='B'):
    """Returns human-readable file size

    Supports:
     * all currently known binary prefixes
     * negative and positive numbers
     * numbers larger than 1000 Yobibytes
     * arbitrary units (maybe you like to count in Gibibits!)

    Example:
    --------
    >>> sizeof_fmt(168963795964)
    '157.4GiB'

    Parameters:
    -----------
    num : int
        Size of file (or data), in bytes

    suffix : str, optional
        Unit suffix, 'B' = bytes by default

    Returns:
    --------
    str
        <floating number><prefix><unit>
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.2f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.2f %s%s" % (num, 'Yi', suffix)
