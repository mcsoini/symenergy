#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mcsoini
"""

import os
import errno
from hashlib import md5
import pandas as pd
from symenergy import _get_logger
import symenergy
from symenergy.auxiliary import params

logger = _get_logger(__name__)

class CacheParams(params.RcParams):
    items_opts = {'path':
                  {'types': {str: 'none'},
                   'values': (),  # only strings, no other value allowed
                   'cond': 'string'}}
    items_default = {'path': os.path.join(list(symenergy.__path__)[0],
                                          'cache')}

cache_params = CacheParams()



def mkdirs(newdir):
    try: os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


class Cache():
    '''
    Handles model cache files.

    Cache files store the model results. They are automatically written
    to pickle files whose filename is generated is generated from a hash
    of the model objects. Existing cache files are automatically read to skip
    the model solution process.
    '''

    prefix = None
    cache_name = 'cache'

    def __init__(self, m_name):
        '''
        Cache instances take the model or evaluator hash as input.
        It is used to generate the filename.

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

        self._m_name = m_name
        self.fn = self.fn_name= None
        self._log_str = None
        self._path = cache_params['path']

        mkdirs(self.path)

        self.fn = self.get_name(self.path)
        sep = os.path.sep
        self.fn_name = (f'...{sep}{self.fn.split(sep)[-2]}{sep}'
                        + os.path.basename(self.fn))

        self._make_log_str()


    @property
    def path(self):

        return self._path

    @path.setter
    def path(self, _):

        raise AttributeError("Attempt to change cache path. Modify "
                             "the symenergy.cache_params['path'] value "
                             "instead, prior to initializing the Model or "
                             "Evaluator class.")


    def _make_log_str(self):

        self._log_str = (f'Loading from cache file {self.fn_name}.',
                        ('Please delete this file to re-solve model: '
                        f'Model.{self.cache_name}.delete()'))


    def load(self):
        '''
        Load model results from cache file.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing model results.
        '''

        smax = len(max(self._log_str, key=len))
        sep_str = ('*' * smax,) * 2
        [logger.warning(st) for st in sep_str + self._log_str + sep_str]

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

    def delete(self):
        ''' Deletes cache file.
        '''

        if os.path.isfile(self.fn):
            logger.info('Removing file %s'%self.fn_name)
            os.remove(self.fn)
        else:
            logger.info('File doesn\'t exist. '
                        'Could not remove %s'%self.fn_name)

    def get_name(self, _dir):
        '''
        Returns a unique hashed model name based on the constraint,
        variable, multiplier, and parameter names.

        Parameters
        ----------
        m : model.Model
           SymEnergy model instance
        '''

        if not self.prefix:
            try:
                prefix_dict = {'cache': 'm', 'cache_eval': 'e', 'cache_lambd': 'l'}
                prefix = prefix_dict[self.cache_name]
            except KeyError as e:
                raise KeyError(str(e) + '. Please define "prefix" class '
                               'attribute in subclassed symenergy cache.')
        else:
                prefix = self.prefix

        m_name = self._m_name[:12].upper()

        fn = f'{prefix}{m_name}.pickle'
        fn = os.path.join(_dir, fn)
        fn = os.path.abspath(fn)

        return fn

    def __repr__(self):

        return (f'Symenergy cache instance id={id(self)}\n'
                f'  * fn =      {self.fn}\n'
                f'  * fn_name = {self.fn_name}\n'
                f'  * file exists: {os.path.isfile(self.fn)}')


class EvaluatorCache(Cache):

    def __init__(self, name, cache_name):

        self.cache_name = cache_name
        super().__init__(name)


    def _make_log_str(self):
        self._log_str = (f'Loading from cache file {self.fn_name}.',
                         ('Please delete this file to re-evaluate: '
                          f'Evaluator.{self.cache_name}.delete()'))

