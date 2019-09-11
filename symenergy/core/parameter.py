#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Parameter class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import sympy as sp

class Parameter():
    '''
    Container class for parameter specification.
    '''

    def __init__(self, name, slot, value):

        self.name = name
        self.slot = slot
        self.value = value

        self.init_symbol()

    def init_symbol(self):

        self.symb = sp.symbols(self.name)

    def __repr__(self):

        return str(self.__class__) + ' ' + self.name

