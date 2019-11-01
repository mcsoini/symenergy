#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains auxiliary functions for parallelization.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import multiprocessing
from multiprocessing import Value, Lock
import numpy as np
import pandas as pd
import itertools
import time
from symenergy import _get_logger

logger = _get_logger(__name__)


class Counter():
    def __init__(self):
        self.val = Value('f', 0)
        self.lock = Lock()

    def reset(self):
        with self.lock:
            self.val.value = 0

    def increment(self):
        with self.lock:
            self.val.value += 1

    def update_ema(self, newval):
        with self.lock:
            if self.val.value == 0:  # first run
                self.val.value = newval
            else:
                self.val.value = self.val.value * 0.99 + 0.01 * newval

    def value(self):
        with self.lock:
            return self.val.value


MP_COUNTER = Counter()
MP_EMA = Counter()


def log_time_progress(f):
    '''
    Decorator for progress logging.
    Note: Due to multiprocessing, this can't be used as an actual decorator.
    https://stackoverflow.com/questions/9336646/python-decorator-with-multiprocessing-fails
    Using explicit wrappers for each method instead.
    '''

    def wrapper(self, df, name, ntot, *args):
        t = time.time()
        res = f(df, *args)
        t = (time.time() - t)  / len(df)

        MP_EMA.update_ema(t)

        vals = (name, int(MP_COUNTER.value()), ntot,
                MP_COUNTER.value()/ntot * 100,
                len(df), MP_EMA.value()/len(df)*1000, t*1000 / len(df))
        logger.info(('{}: {}/{} ({:.1f}%), chunksize {}, '
                     'tavg={:.1f}ms, tcur={:.1f}ms').format(*vals))

        return res
    return wrapper


def get_default_nthreads():

    return multiprocessing.cpu_count() - 1


def parallelize_df(df, func, *args, nthreads='default', concat=True, **kwargs):
    MP_COUNTER.reset()
    MP_EMA.reset()

    if nthreads == 'default':
        nthreads = get_default_nthreads()

    nthreads = min(nthreads, len(df))

    def split(df_):
        nchunks = min(nthreads * 2, len(df_))
        return np.array_split(df, nchunks)

    if isinstance(df, (pd.DataFrame, pd.Series)):
        df_split = split(df)
    elif isinstance(df, list) and isinstance(df[0], (pd.DataFrame, pd.Series)):
        df_split = df
    elif isinstance(df, list):
        df_split = split(df)
    else:
        raise ValueError('Unknown df argument type in parallelize_df')


    pool = multiprocessing.Pool(nthreads)
    if args:
        results = pool.starmap(func, itertools.product(df_split, args))
    else:
        results = pool.map(func, df_split)

    pool.close()
    pool.join()

    if isinstance(results[0], (list, tuple)):
        logger.info('parallelize_df: chaining ... ')
        results = list(itertools.chain.from_iterable(results))

    if isinstance(results[0], (pd.DataFrame, pd.Series)) and concat:
        logger.info('parallelize_df: concatenating ... ')
        results = pd.concat(results)

    logger.info('done.')
    return results



