#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Constraint class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import sympy as sp

class Constraint():

    '''
    Single constraint class.
    Makes sure that:
        * shadow price has correct prefix lb_*
        * sympy constraint expression has correct name cstr_*
        * constraint matches any of the asset.MULTIPS
    '''

    def __init__(self, base_name, slot, multiplier_name='lb',
                 is_equality_constraint=False, var_name=None,
                 is_positivity_constraint=False):
        '''
        Arguments:
            * base_name -- string
            * var_name -- string; the name of the variable the constraint
                          applies to, if applicable
        '''

        self.slot = slot
        self.is_equality_constraint = is_equality_constraint
        self.is_positivity_constraint = is_positivity_constraint
        self.base_name = base_name
        self.multiplier_name = multiplier_name
        self.var_name = var_name

        self._init_shadow_price()
        self._init_column_name()
        self._init_expression()


    @property
    def expr(self):
        if not self._expr:
            raise RuntimeError('Constraint %s: expr undefined'%self.base_name)
        return self._expr

    @expr.setter
    def expr(self, expr):
        if hasattr(expr, 'free_symbols') and self.mlt in expr.free_symbols:
            raise ValueError(('Trying to define constraint %s with expression '
                              'containing multiplier')%self.base_name)

        self.expr_0 = self._expr = expr
        if expr:
            self._expr *= self.mlt


    def _init_shadow_price(self):
        '''
        Sympy symbol
        '''
        self.mlt = sp.symbols('%s_%s'%(self.multiplier_name, self.base_name))


    def _init_expression(self):
        '''
        Component-specific definition of the sympy expression.
        Defined by components. Note that the multiplier is automatically
        included in the expression in the `expr.setter`.
        '''
        self.expr = None


    def _init_column_name(self):
        '''
        The column name used by the Model class to generate the constraint
        combination DataFrame.
        '''

        self.col = 'act_{mult}_{base}'.format(mult=self.multiplier_name,
                                                 base=self.base_name)

    def __repr__(self):

        ret = '%s %s'%(str(self.__class__), self.col)
        return ret



