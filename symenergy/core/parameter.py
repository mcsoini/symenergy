#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Parameter class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import sympy as sp

from symenergy import _get_logger

logger = _get_logger(__name__)

class Parameter():
    '''
    Container class for parameter specification.
    '''

    def __init__(self, name, slot, value):

        self.name = name
        self.slot = slot
        self.value = value

        self._fixed_value = False

        self.init_symbol()

    @property
    def symb(self):
        '''
        Return sympy symbol by default or value if symbol value is fixed.
        '''

        return self._symb if not self._fixed_value else self.value

    @symb.setter
    def symb(self, symb):

        self._symb = symb

    def init_symbol(self):

        self.symb = sp.symbols(self.name)

    def _fix_value(self):

        logger.debug('Fixing value of parameter %s.'%self.name)
        self._fixed_value = True

    def __repr__(self):

        return str(self.__class__) + ' ' + self.name

