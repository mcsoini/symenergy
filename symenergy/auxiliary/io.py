#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mcsoini
"""

import os
import hashlib
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

    def __init__(self, m):
        '''
        Cache instances take the model instance as input. It is solely used
        to generate the hashed filename.

        Parameters
        ----------
        m : model

        Attributes
        ----------
        fn -- str
            Name of cache file
        fn_name -- str
            Shorter cache file name for logging.
        '''

        self.fn = self.get_name(m)

    def load(self):
            fn_name = 'symenergy/cache/' + os.path.basename(self.fn)
            log_str1 = 'Loading from pickle file %s.'%fn_name
            log_str2 = 'Please delete this file to re-solve model.'
            logger.info('*'*max(len(log_str1), len(log_str2)))
            logger.info('*'*max(len(log_str1), len(log_str2)))
            logger.info(log_str1)
            logger.info(log_str2)
            logger.info('*'*max(len(log_str1), len(log_str2)))
            logger.info('*'*max(len(log_str1), len(log_str2)))
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
                        'Could not remove %s'%os.path.abspath(fn))

    def get_name(self, m):
        '''
        Returns a unique hashed model name based on the constraint,
        variable, multiplier, and parameter names.

        Parameters
        ----------
        m : model.Model
           SymEnergy model instance
        '''

        list_slots = ['%s_%s'%(slot.name, str(slot.weight))
                      for slot in m.slots.values()]
        list_slots.sort()
        list_cstrs = [cstr.base_name for cstr in m.constrs]
        list_cstrs.sort()
        list_param = [par.name for par in m.params]
        list_param.sort()
        list_cstrs = [par.name for par in m.variabs]
        list_cstrs.sort()
        list_multips = [par.name for par in m.multips]
        list_multips.sort()

        m_name = '_'.join(list_cstrs + list_param + list_cstrs + list_multips
                          + list_slots)

        m_name = hashlib.md5(m_name.encode('utf-8')).hexdigest()[:12].upper()

        fn = '%s.pickle'%m_name
        fn = os.path.join(list(symenergy.__path__)[0], '..', 'cache', fn)
        fn = os.path.abspath(fn)

        return fn

