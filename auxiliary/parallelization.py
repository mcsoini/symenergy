#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the auxiliary functions.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import multiprocessing
import numpy as np
import pandas as pd
import itertools

def parallelize_df(df, func, nthreads):
    nthreads = min(nthreads, len(df))
    nchunks = min(nthreads, len(df))

    df_split = np.array_split(df, nchunks)
    pool = multiprocessing.Pool(nthreads)
    results = pool.map(func, df_split)
    pool.close()
    pool.join()
    print('parallelize_df: concatenating', end=' ... ')
    if isinstance(results[0], (list, tuple)):
        result = list(itertools.chain.from_iterable(results))
    else:
        result = pd.concat(results)
    print('done.')
    return result
