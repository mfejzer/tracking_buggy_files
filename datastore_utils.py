# -*- coding: utf-8 -*-
"""Utilities to store, save and load features data.

This module is intended to help scripts to be able to easily switch
their output format, provided that they use this module interface to
save data.

Each store format + output format is implemented as a separate class
providing the same API.  Because different formats offer different
capabilities and features, some of those classes offer extra methods /
helpers.

There is also helper function to select appropriate class based on
various input (currently the output filename extension).

NOTE: currently you need to have installed modules for all supported
formats, not only for those used.

Usage:
------
 >>> from datastore_utils import select_datastore_backend
 >>> store = select_datastore_backend(output_filename, category='bug_fixing')
 ...
 >>> store.set_feature(commit, path, feature_name, feature_data)
 ...
 >>> store.print_data(args)

TODO: make interface more generic, removing e.g. args as argument
TODO: use abstract base class equivalent to ensure common API.

"""
from __future__ import print_function

import os
from copy import deepcopy
import pandas as pd

from args_utils import read_json, print_json


class NestedDictStore:
    """Stores data as nested dict, saves it as JSON.

    Uses the following data format:

        {
            "<bugfix commit>": {
                "views": {
                    "<feature category, e.g. 'bug_fixing'>": {
                        "<pathname>": {
                            "<feature name, e.g. 'frequency'>": <feature data>,
                            ...
                        },
                        ...
                    }
                }
            }
        }

    The feature category is set at creation time, in the constructor,
    and defaults to 'bug_fixing'.  The bugfix commit, the pathname,
    and the feature name are given as arguments when storing feature
    data:

     >>> store.set_feature(<bugfix commit>, <pathname>, <feature name>,
                           <feature data>)

    This storage format coupled with output format means that all data
    must be kept in the memory, and it is saved only at time of
    calling print_data() method.  The disadvantage of this is the
    increased memory usage; the advantages are simple and fast adding
    entries to the storage, and minimal filesystem access (low I/O).

    Parameters
    ----------
    filename : str
        Ignored.  Is here to present the same API for all storage
        methods.

        TODO: fix this for consistency, or remove altogeter.

    category : str, optional
        Category under which to put data in JSON.

    data : dict | collections.OrderedDict, optional
        Initial data to be stored, defaults to an empty dict.
        Specific to NestedDictStore class.

    Attributes
    ----------
    data : dict | collections.OrderedDict
        Data that is to be stored, in the format described above.
        May be updated with `store.update(data)`, for example after
        sorting data (and possibly turning it into OrderedDict).

    category : str
        Category / key under which put data in JSON, shown as
        <feature category> in format description.  Defaults to
        'bug_fixing', which is value specific to the first use
        of this module.

    """
    def __init__(self, filename, category='bug_fixing', data=None):
        ## described in class docstring
        if data is not None:
            self.data = data
        else:
            self.data = {}
        self.category = category


    def update(self, data):
        """Update data to value modified outside.

        This may be used for example to make this storage save data to JSON
        sorted, by passing sorted data (in the form of OrderedDict) here;
        the NestedDictStore class doesn't have 'sort' method.

        Parameters
        ----------
        data : dict | collections.OrderedDict
            New data to be stored in NestedDictStore.
        """
        self.data = data


    def set_feature(self, commit, pathname, feature, value):
        """Store f_x(r,s) feature data for given commit/bug and pathname

        In this storage type this method does not impose any
        performance penalty (except of course the cost of calling the
        method).

        Parameters
        ----------
        commit : str
            Shortened SHA-1 of bugfix commit, identifying bug report
            and its fix in the dataset; the $r$ in $f_x(r,s)$.

        pathname : str
            Full pathname relative to the top directory of the project
            of one of files from the project; the $s$ in $f_x(r,s).

        feature : str
            Name of the feature; the $x$ in $f_x(r,s)$.

        value
            The value of the feature $f_x(r,s)$.

        Notes
        -----
        The NestedDictStore takes ownership of the `value` parameter,
        deepcopy-ing it just in case.  It is safe to modify the variable
        passed as the `value` parameter, even if it is a list, a dict
        or other non-scalar object.
        """
        # autovivify commit (not needed if appending to data from JSON)
        if commit not in self.data:
            self.data[commit] = {}
        # autovivify 'views' inner key
        if 'views' not in self.data[commit]:
            self.data[commit]['views'] = {}
        if self.category not in self.data[commit]['views']:
            self.data[commit]['views'][self.category] = {}
        # autovivify pathname
        if pathname not in self.data[commit]['views'][self.category]:
            self.data[commit]['views'][self.category][pathname] = {}

        # set feature; note that value might be array or dict
        self.data[commit]['views'][self.category][pathname][feature] = \
            deepcopy(value)


    def set_from_dict(self, commit, update):
        """Store updated information from dict

        Assumes that dict has the following format:

            {
                "<pathname>": {
                    "<feature data name, e.g. 'frequency'>": <feature data>,
                    ...
                },
                ...
            }

        It also assumes that dict does not share data, but uses either
        fresh or deepcopied data.

        Notes
        -----
        Currently unused and not tested; this was created in an attempt to
        move pick_bug_freq.py v2 code to using datastore_utils module, but
        the script was rewritten to use v3 algorithm and use this module
        instead.

        Parameters
        ----------
        commit : str
            Shortened SHA-1 of bugfix commit, identifying bug report
            and its fix in the dataset; the $r$ in $f_x(r,s)$.

        update : dict | iterable
            New data for multiple features and multiple pathnames for
            given commit/bug, updating current state, replacing the same
            features (using `dict.update`).
        """
        # autovivify commit (not needed if appending to data from JSON)
        if commit not in self.data:
            self.data[commit] = {}
        # autovivify 'views' inner key
        if 'views' not in self.data[commit]:
            self.data[commit]['views'] = {}
        if self.category not in self.data[commit]['views']:
            self.data[commit]['views'][self.category] = {}

        # store updated information
        self.data[commit]['views'][self.category].update(update)


    def read_file(self, filename, preserve_order=False):
        """Read data from JSON file into storage, optionally preserving order

        Data in JSON file should be in prescribed format, but the method
        itself does not check that.

        Parameters
        ----------
        filename : str
            The name of JSON file to read.

        preserve_order : bool, optional
            Whether to preserve order of keys in JSON file, storing data as
            OrderedDict.

        See also
        --------
        NestedDictStore.print_data : saves data in appropriate format
                                     and with appropriate structure
        """
        self.data = read_json(filename, preserve_order)


    def print_data(self, args):
        """Save data to the JSON file (name set by command like arguments)

        Simply dumps data in JSON format using `args_utils.print_json`.  The
        name of the output file (if any, that is if we are not printing to
        standard output), and the formatting is given by command-line
        arguments of the calling script.

        Parameters
        ----------
        args : argparse.Namespace
            Result of running parser.parse_args(), that is expected to
            include options about formatting JSON, results of running
            args_utils.add_json_output_options(parser).  This includes
            args.indent and args.output

        Side effects
        ------------
        Prints information about progress to standard output.
        """
        print_json(self.data, args)



class DataFrameHDF5:
    """Stores data as pandas.DataFrame, saves it as HDF5

    Row index in DataFrame are pathnames.
    Colums in DataFrame are values / features.

    Each commit is saved as separate entry in HDF5.

    Parameters
    ----------
    filename : str
        Name of file in HDF5, opened for appending with pandas.HDFStore()

    category : str, optional
        Ignored.  Is here to present the same API for all storage
        methods.

        TODO: fix this for consistency (title maybe?), or remove altogeter.

    Attributes
    ----------
    store : pandas.HDFStore
        Dict-like IO interface for storing pandas objects in PyTables
        either Fixed or Table format.  Opened for appending.

    df : pandas.DataFrame
        DataFrame holding data for a single commit/bug report, before it is
        saved to `store` under 'g<short sha-1>' key.

    df_br : pandas.DataFrame
        DataFrame holding $br(r,s)$ data for a single commit/bug report $r$
        as a sparse "adjacency" matrix, with value `1` for column $b$ and
        row $s$ if and only if $b$ is in $br(r,s)$.  Saved to `store` under
        'g<short sha-1>/br'

    commit : str | None
        Identifier of the current bugfix commit/bug report.  It is assumed
        that features are stored incrementally, values for one commit after
        all the values for the previous commit.

    """
    # "Learning pandas", chapter "Reading and writing HDF5 format files"
    def __init__(self, filename, category='bug_fixing'):
        ## described in class docstring
        self.df = pd.DataFrame()
        self.df_br = pd.DataFrame()
        self.store = pd.HDFStore(filename)
        self.commit = None


    def _flush_maybe(self, commit):
        """Flush data for single commit from store to HDF5 file, if needed

        If bugfix commit/bug report changed, save current data for previous
        commit in the HDF5 file, making place for storing data for new commit.

        Notes
        -----
        This code was extracted into private method to avoid code
        duplication.

        Parameters
        ----------
        commit : str
            Shortened SHA-1 of bugfix commit, identifying bug report
            and its fix in the dataset
        """
        if self.commit is None:
            self.commit = commit
        elif self.commit != commit:
            # assumes commit by commit order
            # NOTE: 'g' prepended to avoid:
            #
            # /usr/lib/python2.7/dist-packages/tables/path.py:112:
            # NaturalNameWarning: object name is not a valid Python
            # identifier: u'8e6cef0'; it does not match the pattern
            # ``^[a-zA-Z_][a-zA-Z0-9_]*$``; you will not be able to
            # use natural naming to access this object; using
            # ``getattr()`` will still work, though
            self.store['g'+self.commit] = self.df
            if len(self.df_br > 0):
                self.store['g'+self.commit+'/br'] = \
                    self.df_br.to_sparse(fill_value=0)
                # DEBUG
                #print('BR(%s) has density %f' %
                #      (self.commit, self.df_br.to_sparse().density))
            # TODO: make it configurable
            #self.store.flush(fsync=False)
            self.df = pd.DataFrame()
            if len(self.df_br > 0):
                self.df_br = pd.DataFrame()
            self.commit = commit

            # DEBUG
            #print('## %s ===========================================' % commit)


    def set_feature(self, commit, pathname, feature, value):
        """Store f_x(r,s) feature data for given commit/bug and pathname

        If the `commit` parameter is different from previous commit (stored
        in `self.commit` attribute), if flushes DataFrames storing data for
        that previous commit to HDFStore.

        Otherwise it adds (appending single-element DataFrame) or modify
        single entry in `self.df` DataFrame... which is what probably causes
        poor performance of this storage method.

        Parameters
        ----------
        commit : str
            Shortened SHA-1 of bugfix commit, identifying bug report
            and its fix in the dataset; the $r$ in $f_x(r,s)$.

        pathname : str
            Full pathname relative to the top directory of the project
            of one of files from the project; the $s$ in $f_x(r,s).

        feature : str
            Name of the feature; the $x$ in $f_x(r,s)$.

        value
            The value of the feature $f_x(r,s)$.
        """
        self._flush_maybe(commit)

        # unicode in index requires pickling; we can assume sane pathnames
        # NOTE: PyTables for Python 2.x specific
        pathname = str(pathname)

        if feature not in self.df.columns:
            # if column does not exist, add it
            self.df[feature] = pd.Series({pathname: value})
        elif pathname not in self.df.index:
            # if row does not exist, add it
            self.df.append(
                pd.DataFrame.from_dict({feature: {pathname: value}})
            )
        else:
            # modify scalar value
            self.df.loc[pathname, feature] = value


    def set_br(self, data, commit, pathname, feature, br_list):
        """Store br(r,s) with value being a list of commits

        The list of bugfix commits/bug reports is stored in "adjacency"
        matrix for given `commit`, where index/row is `pathname` and the
        DataFrame has ones for columns being values of `br_list`.

        This is then turned into SparseDataFrame for storage, and saved
        under 'g<commit>/<feature>' key in HDF5 file.

        Notes
        -----
        The idea was that sparse matrix would be stored effectively
        compressed in HDF5 file.  It turns out that it is not so; this way
        of storing br(r,s) data takes up very large amound of disk space;
        repacking doesn't help here.

        Parameters
        ----------
        commit : str
            Shortened SHA-1 of bugfix commit, identifying bug report
            and its fix in the dataset; the $r$ in $f_x(r,s)$.

        pathname : str
            Full pathname relative to the top directory of the project
            of one of files from the project; the $s$ in $f_x(r,s).

        feature : str
            Name of the feature; the $x$ in $f_x(r,s)$.  Would be 'br' for
            feature $br(r,s)$, for example.

        br_list : list of str
            List of commits (of shortened sha-1 ideintifiers of bugfix
            commits) that are values of $f_x(r,s)$
        """
        self._flush_maybe(commit)

        # do not add empty list
        if len(br_list) == 0:
            return

        # unicode in index requires pickling; we can assume sane pathnames
        # NOTE: PyTables for Python 2.x specific
        pathname = str(pathname)

        # one row dataframe
        df_new = pd.DataFrame({pathname: pd.Series(
            [1]*len(br_list), index=[br_list],
        )}).T

        # DEBUG
        #print('df_new')
        #print(df_new)
        # DEBUG
        #print('df_br')
        #print(self.df_br)

        if pathname not in self.df_br.index:
            # if row does not exist, add it
            # append() is special case of concat()
            self.df_br = self.df_br.append(df_new)
        else:
            # if it exists, modify it (join on index)
            # join() is special case of merge()
            self.df_br = self.df_br.join(df_new)


    def read_file(self, filename):
        """Read data from HDF5 file into storage, saving current data

        Ensures that all data is saved, and flushes it to filesystem, but it
        currently does not explicitely close the previous file
        (`self.store`).

        Parameters
        ----------
        filename : str
            The name of HDF5 file to read.

        See also
        --------
        DataFrameHDF5.print_data : saves data in appropriate format
                                   and with appropriate structure
        """
        # save old data
        self.store['g'+self.commit] = self.df
        self.store.flush(fsync=True)
        # read new
        self.store = pd.HDFStore(filename)
        self.commit = None


    def print_data(self, args):
        """Save data to the HDF5 file (given at store creation time)

        Actually ensures that all data is put into PyTable (in HDFStore),
        and that data is flushed to disk (fsync-ed).  Does not close
        `self.store`.

        Note that contrary to JSON format, HDF5 is a binary format and is
        not suitable for printing to standard output.

        Parameters
        ----------
        args : argparse.Namespace
            Ignored currently.
        """
        # store what may have been not stored
        self.store['g'+self.commit] = self.df
        # ensure that it is written to disk
        self.store.flush(fsync=True)



def select_datastore_backend(filename, category='bug_fixing'):
    """Given file name, return appropriate storage object

    This function allows scripts using this module to select mechanism for
    storing feature data.  Currently the choice is based on the output file
    pathname, or to be more exact its extension.

    Examples
    --------
     >>> from datastore_utils import select_datastore_backend
     ...
     >>> store = select_datastore_backend('test.json', category='bug_fixing')
     >>> store
     <datastore_utils.NestedDictStore instance at 0x7fe9e16ed200>

    Parameters
    ----------
    filename : str | unicode
        Name of the filename where to save collected features data.

    category : str, default 'bug_fixing'
        Category characterizing the set of features that would be saved in
        the store.

    Returns
    -------
    NestedDictStore | DataFrameHDF5
        Currently returns DataFrameHDF5 if `filename` has '.h5' extension,
        NestedDictStore if filename has '.json' extension and in all other
        cases.
    """
    if filename is None:
        return NestedDictStore(filename, category)

    (root, ext) = os.path.splitext(filename)

    if ext == '.json':
        return NestedDictStore(filename, category)
    elif ext == '.h5':
        return DataFrameHDF5(filename, category)
    else:
        return NestedDictStore(filename, category)

# end of datastore_utils.py
