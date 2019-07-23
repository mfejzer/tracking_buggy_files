#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities to process datasets read from JSON.

All subroutines / functions should be free from side-effects, and do
not print anything on standard output or standard error.

Assumes the following structure of data in dataset (for `data`
arguments); note that the structure of commit metadata describes
original pre-fix state:

  {
    "<7-char shortened SHA-1 identifier of bugfix commit; e.g. deadbee>": {
        "bug_report": {
            "id": "<entry index, values from 1 to number of entries; e.g.: 177>",
            "bug_id": "<bug identifier from bugtracker, a number; e.g.: 155148",
            "timestamp": "<bug report creation date, as timestamp; e.g.: 1156470000>",
            "summary": "<one line description of bug report>",
            "description": "<bug report; e.g.: multiple lines\nseparated with\nnewline>",
            "status": "<final status of bug report; e.g.: resolved fixed>",
            "commit": "<shortened SHA-1 identifier of bugfix, same as entry key; e.g. deadbee>",
            "result": "<Learn to Rank results; e.g.: 80:path/to/file\n333:path/to/other/file>"
        },
        "commit": {
            "metadata": {
                "sha": "commit deadbeefa704548a42e396e996c9d49915b92a64\n",
                "author": "Author: Joe Hacker <joe@example.com>\n",
                "date": "Date:   Fri Aug 25 14:37:34 2006 +0000\n",
                "message": "multi line\n commit message\n"
            },
            "diff": {
                "path/to/file": "diff --git ...\n...\n"
            }
        }
    },
    ...
  }

"""
from collections import OrderedDict
from bitmap import BitMap
import array


def sorted_by_bugreport(data, key='timestamp'):
    """Return collections.OrderedDict sorted by bug report params

    Should be used either with key='timestamp' (the default),
    or with key='bug_id'.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    key : string, optional
        One of keys in the 'bug_report' inner dict.

    Returns
    -------
    OrderedDict
        Input data, sorted.
    """
    return OrderedDict(
        sorted(data.items(),
               key=lambda t: int(t[1]['bug_report'][key])
        )
    )


def sorted_by_commit(data, key='timestamp'):
    """Return collections.OrderedDict sorted by commit metadata

    Should be used either with key='timestamp' (the default),
    or with key='author'; sorting by 'sha' is also possible,
    but it doesn't make much sense.

    Note that the key needs to exist.  You can ensure that
    the 'timestamp' exist with `fix_commit_metadata()`.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    key : string, optional
        One of keys in the metadata for fixup commit.

    Returns
    -------
    OrderedDict
        Input data, sorted.
    """
    return OrderedDict(
        sorted(data.items(),
               key=lambda t: int(t[1]['commit']['metadata'][key])
        )
    )


def bugfix_to_idx(data, commits):
    """Convert shortened SHA-1 of bugfix commit to 'id' field (entry number)

    NOTE: Currently there is no any error handling at all.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    commits : string | list
        SHA-1 of the bugfix commit, shortened to 7 characters, or list
        of such values.  Shortened SHA-1 of bugfix is used as key in
        the `data` dict.

    Returns
    -------
    int | list
        Value of the 'id' field for given bug report, which is
        the index of the entry, or sorted list of such values.
    """
    if isinstance(commits, list):
        return sorted([ bugfix_to_idx(data, c) for c in commits ])
    else:
        return int(data[commits]['bug_report']['id'])


def idx_to_bugfix(data, ids, bugfix_list=[]):
    """Convert 'id' field (entry number) to shortened SHA-1 of bugfix commit

    NOTE: Currently there is no any error handling at all.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    ids : int | list
        Value of the 'id' field for given bug report, which is
        the index of the entry, or list of such values.

    bugfix_list : list
        List of bugfix commits, where each bugfix is at index given by
        its 'id' value.

        NOTE: this parameter is (ab)used as static variable.

    Returns
    -------
    int | list
        SHA-1 of the bugfix commit, shortened to 7 characters, or list
        of such values.  Shortened SHA-1 of bugfix can be used as key
        in the `data` dict.

        idx_to_commit(commit_to_idx(c)) == c, if c is shortened SHA-1
        identifier of the bugfix commit.
    """
    if isinstance(ids, list):
        return [ idx_to_bugfix(data, idx) for idx in ids ]

    if not bugfix_list:
        bugfix_list.extend([None] * (len(data)+1))
        for commit in data:
            bugfix_list[ bugfix_to_idx(data, commit) ] = commit

    if ids in bugfix_list:
        return bugfix_list[ids]

    # should never happen
    for commit in data:
        idx = bugfix_to_idx(data, commit)
        if idx == ids:
            return commit

    # ids not found
    return None


def list_of_ids_to_bitmap(ids, maxitems, bitmap=None):
    """Convert list of 'id' values (entry numbers) to bitmap / bitset

    Each 'id' value is an entry number in the original dataset, that
    is the 'id' column in the database.  Those have are natural
    numbers, and have values between 1 and number of entries; the
    latter is given as parameter to this function.

    Returned bitmap has i-th bit set to "1" (has "1" at i-th place) if
    and only if there was identifier 'i' on the list.

    Parameters
    ----------
    ids | list
        [Sorted] list of integers with values between 1 and
        `maxitems`, inclusive; those are 'id' fields for given bug
        report / bugfix commit.

    maxitems : int
        Maximum value of ids onn the list, which is number of entries
        in the dataset; this means that it is the minimal number of
        bits in the bitmap / bitset.

    bitmap : BitMap, optional
        Bitmap object to set, in current incarnation in needs
        .set(i-th) method to set i-th bit to "1" in resulting bitmap.
        The bitmap must be empty (all zeros), and have at least
        `maxitems` bits.

    Returns
    -------
    bitmap.BitMap
        Bitmap with appropriate bits set.  Note that values are from 1
        to maxitems, while bit positions are numbered from 0 to
        maxitems-1.

    """
    if not bitmap:
        bitmap = BitMap(maxitems)

    # there is no built-in initialization from iterable for bitmap.BitMap
    for i in ids:
        # ids are numbered from 1, bits are numbered from 0
        bitmap.set(i-1)

    return bitmap


def bitmap_to_list_of_ids(bitmap):
    """Convert bitmap to list of values (from 1 to number of bits)

    Returns list of indices of non-zero bits in the given bitmap,
    counting positions from 1.  This function is the reverse of
    list_of_ids_to_bitmap().

    Parameters
    ----------
    bitmap : BitMap
        Bitmap / bitset denoting 'id' in set.

    Returns
    -------
    list
        List of 'id' fields (entry numbers).  It is sorted list of
        integers, each value between 1 and maxitems.

    """
    # ids are numbered from 1, bits are numbered from 0
    return [ (i+1) for i in bitmap.nonzero() ]


def bitmap_to_bytes(bitmap):
    """Turn bitmap into string of bytes

    The returned value is representation of bitmap suitable for store
    in the DataFrame, and ultimately in the HDF5 file.

    Parameters
    ----------
    bitmap : BitMap
        Bitmap / bitset, which representation we want to get.

    Returns
    -------
    bytes (str in Python 2.x)
        [Compact] representation of bitmap as bytes.  Note that it may
        contain NUL ('\x00') characters.

    """
    return bitmap.bitmap.tostring()


def bytes_to_bitmap(buf, maxitems, bitmap=None):
    """Convert string of bytes back into bitmap

    This is the inverse of bytes_to_bitmap(), and is used to recover
    the bitmap from its representation as string of bytes.

    Parameters
    ----------
    buf : bytes (str in Python 2)
        [Compact] representation of bitmap as bytes.  Result of
        bitmap_to_bytes().

    maxitems : int
        Number of entries; number of bits in bitmap.

    bitmap : BitMap, optional
        Bitmap object to set.  If provided, the bitmap must be empty
        (all zeros), and have at least `maxitems` bits.  This can be
        used to avoid commit creation costs.

    Returns
    -------
    BitMap
        Bitmap / bitset with given representation.
    """
    if not bitmap:
        bitmap = BitMap(maxitems)

    bitmap.bitmap = array.array('B', buf)
    return bitmap


def list_of_bugfixes_to_storage(data, commits):
    """Turn list of bugfixes into form suitable for storage [in HDF5]

    Given list of bugfix commit identifiers (keys to the bug
    report+fix info in the dataset), turn it into something suitable
    for storage as value in DataFrame, that can be stored without
    pickling in HDF5 -- for example string of bytes.

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    commits : list
        List of shortened SHA-1 identifiers of bugfix commits, which
        are keys in `data` dict; represents set of bug reports and
        their fixes.

    Returns
    -------
    bytes (str in Python 2.x)
        Compact representation of given list (set) of commits,
        well suited for storage.
    """
    return bitmap_to_bytes(
        list_of_ids_to_bitmap(bugfix_to_idx(data, commits),
                              len(data))
    )


def storage_to_list_of_bugfixes(data, buf):
    """Recover list of bugfix commits from storage representation

    This function is the inverse of list_of_bugfixes_to_storage(), as
    one might have expected.

    It turns compact representation of set (list) of bugfix commits
    from storage (for example from DataFrame stored in HDF5 file) into
    actual list of bugfix commit identifiers (shortened SHA1s).

    Parameters
    ----------
    data : dict | OrderedDict
        The combined bug report and repository information from
        the JSON file.

    buf : bytes (str in Python 2)
        Compact and fast representation of list of bugfix commits
        taken from storage.

    Returns
    -------
    list
        List of shortened SHA-1 identifiers of bugfix commits, which
        are keys in `data` dict; represents set of bug reports and
        their fixes.

    """
    bitmap = bytes_to_bitmap(buf, len(data))
    ids = bitmap_to_list_of_ids(bitmap)
    return idx_to_bugfix(data, ids)


# end of file dataset_utils.py
