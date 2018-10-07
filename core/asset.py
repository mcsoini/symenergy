#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:18:18 2018

@author: user
"""

import sympy as sp

import symenergy.core.component as component
from symenergy.core.constraint import Constraint
from symenergy.core.slot import noneslot



class UnexpectedSymbolError(Exception):

    def __init__(self, res):

        message = ('Method called with unexpected '
                   'variable/parameter %s'%(res))

        super().__init__(message)

class Asset(component.Component):
    '''Mixin class containing shared methods of plants and storage'''

    # capacity class map
    MAP_CAPACITY = {'C': 'p',   # all power
                    'E': 'e'}   # storage energy capacity

    def __init__(self):

        self.params = []

    def init_cstr_capacity(self, capacity_name):
        '''
        Instantiates a dictionary {slot symbol: Constraint}.
        '''

        if not capacity_name in self.MAP_CAPACITY.keys():
            raise UnexpectedSymbolError(capacity_name)


        if self.MAP_CAPACITY[capacity_name] in self.VARIABS_TIME:
            slot_objs = self.slots.values()
        elif self.MAP_CAPACITY[capacity_name] in self.VARIABS:
            slot_objs = [noneslot]

        setattr(self, 'cstr_cap_%s'%capacity_name, {})
        cstr_dict = getattr(self, 'cstr_cap_%s'%capacity_name)

        for slot in slot_objs:

            base_name = '%s_cap_%s_%s'%(self.name, capacity_name,
                                     str(slot.name))

            cstr = Constraint(base_name=base_name, slot=slot)

            # define expression
            var = getattr(self, self.MAP_CAPACITY[capacity_name])[slot]
            cap = getattr(self, capacity_name).symb

            # subtract retired capacity if applicable
            if capacity_name + '_ret' in self.__dict__:
                cap -= getattr(self, capacity_name + '_ret')[noneslot]

            cstr.expr = cstr.mlt * (var - cap)

            cstr_dict[slot] = cstr

    def init_cstr_positive(self, variable):
        '''
        Instantiates a dictionary {slot symbol: Constraint}.
        '''

        if variable in self.VARIABS_TIME:
            slot_objs = self.slots.values()
        elif variable in self.VARIABS:
            slot_objs = [noneslot]
        else:
            raise UnexpectedSymbolError(variable)

        setattr(self, 'cstr_pos_%s'%variable, {})
        cstr_dict = getattr(self, 'cstr_pos_%s'%variable)

        for slot in slot_objs:

            base_name = '%s_pos_%s_%s'%(self.name, variable, str(slot.name))

            cstr = Constraint(base_name=base_name, slot=slot)

            var = getattr(self, variable)[slot]
            cstr.expr = cstr.mlt * var

            cstr_dict[slot] = cstr

    def init_is_capacity_constrained(self, capacity_name, variable):

        self.is_capacity_constrained = (getattr(self, variable)
                                        if '%s_%s'%(capacity_name, self.name)
                                        in self.get_params_dict('name')
                                        else None)

    def init_is_positive(self):
        ''''''
        self.is_positive = [var
                            for variab in self.VARIABS_POSITIVE
                            if hasattr(self, variab)
                            for _, var in getattr(self, variab).items()]

    def init_symbol_operation(self, variable):
        '''
        Sets operational variables, i.e. power (generation,
        charging, discharging) and stored energy.
        '''

        if variable in self.VARIABS:
            setattr(self, variable,
                    {noneslot: sp.symbols('%s_%s_%s'%(self.name,
                                                  variable,
                                                  str(None)))})

        elif variable in self.VARIABS_TIME:
            setattr(self, variable,
                    {slot: sp.symbols('%s_%s_%s'%(self.name,
                                                  variable,
                                                  str(slot.name)))
                     for slot in self.slots.values()})
        else:
            raise UnexpectedSymbolError(variable)

    def init_symbols_costs(self):
        ''' Overridden by children, if applicable. '''

    def _subs_cost(self, symb, *args, **kwargs):
        ''' Overridden by children, if applicable. '''

        return symb

    def __repr__(self):

        return '%s %s'%(self.__class__, str(self.name))


