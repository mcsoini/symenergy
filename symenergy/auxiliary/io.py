#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:54:26 2019

@author: mcsoini
"""

import os
import hashlib
import pandas as pd
from symenergy import _get_logger
import symenergy

logger = _get_logger(__name__)

class Cache():

    def __init__(self, m):

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

        df.to_pickle(self.fn)


    @property
    def file_exists(self):

        return os.path.isfile(self.fn)

    @staticmethod
    def delete_cached(fn):
        if os.path.isfile(fn):
            logger.info('Removing file %s'%os.path.abspath(fn))
            os.remove(fn)
        else:
            logger.info('File doesn\'t exist. '
                        'Could not remove %s'%os.path.abspath(fn))

    def get_name(self, m):
        '''
        Returns a unique hashed model name based on the constraint names.
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

