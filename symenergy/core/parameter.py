#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Parameter class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import sympy as sp

from symenergy.auxiliary.decorators import hexdigest
from symenergy import _get_logger

logger = _get_logger(__name__)

class Parameter():
    '''
    Container class for parameter specification.
    '''

    def __init__(self, base_name, slot, value):

        self._is_frozen = False

        self.slot = slot

        self.base_name = base_name
        self.name = '%s_%s'%(base_name, slot.name)
        self.value = value

        self.init_symbol()


    @property
    def symb(self):
        '''
        Return sympy symbol by default or value if symbol value is fixed.
        '''

        return self._symb if not self._is_frozen else self.value


    @symb.setter
    def symb(self, symb):

        self._symb = symb


    @property
    def value(self):

        return self._value


    @value.setter
    def value(self, val):

        if self._is_frozen:
            raise RuntimeError('Trying to redefine value of frozen parameter '
                               '%s with current value %s'%(self.name,
                                                           self.value))
        else:
            self._value = val


    def init_symbol(self):

        self.symb = sp.symbols(self.name)


    def _freeze_value(self):

        logger.debug('Fixing value of parameter %s.'%self.name)
        self._is_frozen = True


    def _unfreeze_value(self):

        logger.debug('Unfreezing value of parameter %s.'%self.name)
        self._is_frozen = False


    @hexdigest
    def _get_hash_name(self):
        '''
        The parameter hash includes the parameter value only if it is frozen.
        '''

        return self.name + (str(self.value) if self._is_frozen else '')


    def __repr__(self):

        return str(self.__class__) + ' ' + self.name

