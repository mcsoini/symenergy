#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Parameter class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import re
import sympy as sp

from symenergy import _get_logger

logger = _get_logger(__name__)

class Variable():
    '''
    Container class for variable specification.
    '''

    def __init__(self, base_name, slot, comp_name):

        self.slot = slot
        self.base_name = base_name
        self.comp_name = comp_name
        self.name = '%s_%s_%s'%(comp_name, base_name, slot.name)
        self.name_no_comp = re.sub(self.comp_name + '_', '', self.base_name)

        self.init_symbol()


    def init_symbol(self):

        self.symb = sp.symbols(self.name)


    def __repr__(self):

        return str(self.__class__) + ' ' + self.name

