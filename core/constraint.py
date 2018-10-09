#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Constraint class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import sympy as sp

class Constraint():

    '''
    Constructs constraints.
    Makes sure that:
        * shadow price has correct prefix lambda_*
        * sympy constraint expression has correct name cstr_*
        * constraint matches any of the asset.MULTIPS
    '''

    def __init__(self, base_name, slot, multiplier_name='lb',
                 is_equality_constraint=False):
        '''
        Arguments:
            * base_name -- string
        '''

        self.slot = slot
        self.is_equality_constraint = is_equality_constraint
        self.base_name = base_name
        self.multiplier_name = multiplier_name

        self.init_shadow_price()
        self.init_column_name()
        self.init_expression()


    @property
    def expr(self):
        if not self.__expr:
            raise ValueError('Constraint %s: expr undefined'%self.base_name)
        return self.__expr

    @expr.setter
    def expr(self, expr):
        self.__expr = expr



    def init_shadow_price(self):
        '''
        Sympy symbol
        '''
        self.mlt = sp.symbols('%s_%s'%(self.multiplier_name, self.base_name))


    def init_expression(self):
        '''
        Component-specific definition of the sympy expression.
        Implemented there.
        '''
        self.expr = None


    def init_column_name(self):
        '''
        The column name used by the Model class to generate the constraint
        combination DataFrame.
        '''

        self.col = 'act_{mult}_{base}'.format(mult=self.multiplier_name,
                                                 base=self.base_name)

    def __repr__(self):

        ret = '%s %s'%(str(self.__class__), self.col)
        return ret



