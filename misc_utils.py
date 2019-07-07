# -*- coding: utf-8 -*-
"""Miscelanous utilities that do not fit other modules.

This module is meant to include generic functions and classes, usually
taken from somewhere, that could conceveiably be used by more than one
script, but do not fit cleanly into any other module.

"""
from __future__ import print_function

# "Compute Memory footprint of an object and its contents « Python recipes « ActiveState Code"
# https://code.activestate.com/recipes/577504/
#
# https://github.com/ActiveState/code
# code/recipes/Python/577504_Compute_Memory_footprint_object_its/recipe-577504.py
#from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass


def total_size(obj, handlers={}, verbose=False):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    Examples
    --------
     >>> from misc_utils import total_size
     ...
     >>> d = dict(a=1, b=[2,3,4], c='a string of chars')
     >>> print(total_size(d, verbose=False))
     640

    Parameters
    ----------
    obj : obj
        Object to find memory footprint of.

    handlers : dict, optional
        Handlers to iterate over contents of container classes; keys
        are container types, values are functions returning list of
        elements in a container.  There is built-in support for tuple,
        list, deque, dict, set and frozenset.

    verbose : bool, default False
        Whether to print progress reports to sys.stderr.

    Returns
    -------
    int
        Memory footprint of object in bytes.
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(obj):
        if id(obj) in seen:       # do not double count the same object
            return 0
        seen.add(id(obj))
        s = getsizeof(obj, default_size)

        if verbose:
            print(s, type(obj), repr(obj), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(obj, typ):
                s += sum(map(sizeof, handler(obj)))
                break
        return s

    return sizeof(obj)


# https://stackoverflow.com/questions/38545828/pandas-describe-by-additional-parameters
def describe_extra(df, stats):
    """Add extra stats to Pandas DataFrame's describe()

    Examples
    --------
     >>> import pandas as pd
     >>> from misc_utils import describe_extra
     ...
     >>> df = pd.DataFrame([[0, 0, 0],\
                            [0, 1, 0],\
                            [0, 2, 0],\
                            [0, 3, 6]], columns=['zeros', 'range', 'outlier'])
     ... # mad  = Median Absolute Deviation, mad(x_i) = median(|x_i - median(x_i)|)
     ... # skew = skewness, a measure of the asymmetry of the distribution about its mean
     ... # kurt = kurtosis, a measure of the "tailedness" of the distribution
     >>> describe_extra(df, ['mad', 'skew', 'kurt'])
            zeros     range  outlier
     count    4.0  4.000000     4.00
     mean     0.0  1.500000     1.50
     std      0.0  1.290994     3.00
     min      0.0  0.000000     0.00
     25%      0.0  0.750000     0.00
     50%      0.0  1.500000     0.00
     75%      0.0  2.250000     1.50
     max      0.0  3.000000     6.00
     mad      0.0  1.000000     2.25
     skew     0.0  0.000000     2.00
     kurt     0.0 -1.200000     4.00

    Notes
    -----
    misc_utils.py:83: FutureWarning: '.reindex_axis' is deprecated and
    will be removed in a future version. Use '.reindex' instead.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to find exended describe() of.
        TODO: pass result of df.describe() instead, for better control.

    stats : list of str
        List of names of statistics functions supported by DataFrame.

    Returns
    -------
    pandas.Series | pandas.DataFrame
        Series/DataFrame of summary statistics, augmented with given
        list of extra stats.
    """
    d = df.describe()
    return d.append(df.reindex_axis(d.columns, 1).agg(stats))


def cmp_char(a, b):
    """Returns '<', '=', '>' depending on whether a < b, a = b, or a > b

    Examples
    --------
     >>> from misc_utils import cmp_char
     >>> cmp_char(1, 2)
     '<'
     >>> print('%d %s %d' % (1, cmp_char(1,2), 2))
     1 < 2

    Parameters
    ----------
    a
        Value to be compared
    b
        Value to be compared

    Returns
    -------
    {'<', '=', '>'}
        Character denoting the result of comparing `a` and `b`.
    """
    if a < b:
        return '<'
    elif a == b:
        return '='
    elif a > b:
        return '>'
    else:
        return '?'
