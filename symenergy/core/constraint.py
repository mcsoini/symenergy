#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Constraint class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import re
import sympy as sp

#
#class ConstraintCollection(list):
#
#    def __init__(self):
#        pass
#
#
#    def
#
#
#cc = ConstraintCollection()
#
#cc.append('a')
#cc.append('b')
#cc

class Constraint():

    '''
    Single constraint class.
    Makes sure that:
        * shadow price has correct prefix lb_*
        * sympy constraint expression has correct name cstr_*
        * constraint matches any of the asset.MULTIPS
    '''

    dict_mult_name = {True: 'pi',  # equality constraints
                      False: 'lb'}  # inequality constraints


    def __init__(self, base_name, slot, expr_func,
                 is_equality_constraint=False, var_name=None,
                 is_positivity_constraint=False,
                 is_capacity_constraint=False, comp_name='model',
                 expr_args=tuple()):
        '''
        Arguments:
            * base_name -- string
            * var_name -- string; the name of the variable the constraint
                          applies to, if applicable
        '''

        self.slot = slot
        self._expr_func = expr_func

        self.is_positivity_constraint = is_positivity_constraint
        self.is_equality_constraint = is_equality_constraint
        self.is_capacity_constraint = is_capacity_constraint

        self.multiplier_name = self.dict_mult_name[is_equality_constraint]

        self.base_name = base_name
        self.var_name = var_name
        self.name = '%s_%s' % (base_name, slot.name)

        if not is_equality_constraint:
            assert comp_name, ('No comp_name supplied to constraint %s. '
                               'Inequality constraints must be associated '
                               'with a component.') % self.name

        self.comp_name = comp_name

        self.name_no_comp = None
        if self.comp_name:
            self.name_no_comp = re.sub(self.comp_name + '_', '', self.base_name)

        self.expr_args = expr_args

        self._init_shadow_price()
        self._init_column_name()
        self.make_expr()


    def make_expr(self):
        '''
        Uses the expression function to generate the expression.
        '''
        self.expr = self._expr_func(*self.expr_args)


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
        self.mlt = sp.symbols('%s_%s_%s'%(self.multiplier_name, self.base_name,
                                          self.slot.name))


    def _init_column_name(self):
        '''
        The column name used by the Model class to generate the constraint
        combination DataFrame.
        '''

        name_dict = dict(mult=self.multiplier_name, base=self.base_name,
                         slot=self.slot.name)
        self.col = 'act_{mult}_{base}_{slot}'.format(**name_dict)


    def __repr__(self):

        ret = '%s %s'%(str(self.__class__), self.col)
        return ret



