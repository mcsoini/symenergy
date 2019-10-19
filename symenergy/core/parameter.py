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

        self._frozen_value = False

        self.name = name
        self.slot = slot
        self.value = value

        self.init_symbol()


    @property
    def symb(self):
        '''
        Return sympy symbol by default or value if symbol value is fixed.
        '''

        return self._symb if not self._frozen_value else self.value


    @symb.setter
    def symb(self, symb):

        self._symb = symb


    @property
    def value(self):

        return self._value


    @value.setter
    def value(self, val):

        if self._frozen_value:
            raise RuntimeError('Trying to redefine value of frozen parameter '
                               '%s with current value %s'%(self.name,
                                                           self.value))
        else:
            self._value = val


    def init_symbol(self):

        self.symb = sp.symbols(self.name)


    def _freeze_value(self):

        logger.debug('Fixing value of parameter %s.'%self.name)
        self._frozen_value = True


    def __repr__(self):

        return str(self.__class__) + ' ' + self.name

