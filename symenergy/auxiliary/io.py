#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mcsoini
"""

import os
from hashlib import md5
import pandas as pd
from symenergy import _get_logger
import symenergy

logger = _get_logger(__name__)

class Cache():
    '''
    Handles model cache files.

    Cache files store the model results. They are automatically written
    to pickle files whose filename is generated is generated from a hash
    of the model objects. Existing cache files are automatically read to skip
    the model solution process.
    '''

    def __init__(self, m_name):
        '''
        Cache instances take the model instance as input. It is solely used
        to generate the hashed filename.

        Parameters
        ----------
        m_name : str

        Attributes
        ----------
        fn -- str
            Name of cache file
        fn_name -- str
            Shorter cache file name for logging.
        '''

        self.fn = self.get_name(m_name)
        self.fn_name = 'symenergy/cache/' + os.path.basename(self.fn)

    def load(self):
        '''
        Load model results from cache file.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing model results.
        '''

        log_str = ('Loading from cache file %s.'%self.fn_name,
                   ('Please delete this file to re-solve model: '
                   'Model.cache.delete_cached()'))
        smax = len(max(log_str, key=len))
        sep_str = ('*' * smax,) * 2
        [logger.info(st) for st in sep_str + log_str + sep_str]

        return pd.read_pickle(self.fn)


    def write(self, df):
        ''' Write dataframe to cache file.

        Parameters
        ----------
        df : pandas.DataFrame
            Table with model results
        '''

        df.to_pickle(self.fn)


    @property
    def file_exists(self):
        ''' Checks whether the cache file exists.

        Returns
        -------
        bool
            True if the cache file corresponding to the hashed filename exists.
            False otherwise.
        '''

        return os.path.isfile(self.fn)

    def delete_cached(self):
        ''' Deletes cache file.
        '''

        if os.path.isfile(self.fn):
            logger.info('Removing file %s'%self.fn_name)
            os.remove(self.fn)
        else:
            logger.info('File doesn\'t exist. '
                        'Could not remove %s'%self.fn_name)

    def get_name(self, m_name):
        '''
        Returns a unique hashed model name based on the constraint,
        variable, multiplier, and parameter names.

        Parameters
        ----------
        m : model.Model
           SymEnergy model instance
        '''

        m_name = m_name[:12].upper()

        fn = '%s.pickle'%m_name
        fn = os.path.join(list(symenergy.__path__)[0], '..', 'cache', fn)
        fn = os.path.abspath(fn)

        return fn

