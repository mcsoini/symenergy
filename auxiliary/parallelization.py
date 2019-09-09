#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains auxiliary functions for parallelization.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import multiprocessing
import numpy
import pandas as pd
import itertools
from symenergy import _get_logger

logger = _get_logger(__name__)

try:
    from pathos.multiprocessing import ProcessingPool as Pool
except Exception as e: logger.info(e)

def parallelize_df(df, func, nthreads, use_pathos=False, **kwargs):
    nthreads = min(nthreads, len(df))
    nchunks = min(nthreads * 2, len(df))

    df_split = numpy.array_split(df, nchunks)
    if use_pathos:
        pool = Pool(nthreads)
        results = pool.map(func, df_split, **kwargs)
    else:
        pool = multiprocessing.Pool(nthreads)
        results = pool.map(func, df_split)
    pool.close()
    pool.join()
    if use_pathos:
        pool.clear()
        pool.restart()
    logger.info('parallelize_df: concatenating ... ')
    if isinstance(results[0], (list, tuple)):
        result = list(itertools.chain.from_iterable(results))
    else:
        result = pd.concat(results)
    logger.info('done.')
    return result


