#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the auxiliary functions.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import multiprocessing
import numpy as np
import pandas as pd

def parallelize_df(df, func, nthreads):
    nthreads = min(nthreads, len(df))
    nchunks = min(nthreads * 4, len(df))

    df_split = np.array_split(df, nchunks)
    pool = multiprocessing.Pool(nthreads)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
