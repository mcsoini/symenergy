#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:18:18 2018

@author: user
"""

import sympy as sp

import symenergy.core.component as component
from symenergy.core.constraint import Constraint
from symenergy.core.parameter import Parameter
from symenergy.core.slot import noneslot



class UnexpectedSymbolError(Exception):

    def __init__(self, res):

        message = ('Method called with unexpected '
                   'variable/parameter %s'%(res))

        super().__init__(message)

class Asset(component.Component):
    '''Mixin class containing shared methods of plants and storage'''

    # capacity class map
    MAP_CAPACITY = {'C': ['p', 'C_ret'],   # all power and retired capacity
                    'E': ['e']}   # storage energy capacity

    def __init__(self):

        self.params = []


    def get_constrained_variabs(self):
        '''
        Keyword argument:
            * cap -- string, capacity name as specified in self.MAP_CAPACITY

        Returns:
            * list of tuples [(parameter object capacity,
                               list relevant var strings)]
        '''

        cap_var = []

        c_name, variabs = list(self.MAP_CAPACITY.items())[0]
        for c_name, variabs in self.MAP_CAPACITY.items():
            if hasattr(self, c_name):
                cap = getattr(self, c_name)

                constr_var = []

                for var in variabs:
                    if hasattr(self, var):
                        variabs = getattr(self, var, None)

                        constr_var += list(variabs.values())

                cap_var.append((cap, constr_var))

        return cap_var

    def init_cstr_capacity(self, capacity_name):
        '''
        Instantiates a dictionary {slot symbol: Constraint}.

        Applies to power and capacity retirement, both of which are smaller
        than the initially installed capacity.
        '''

        if not capacity_name in self.MAP_CAPACITY:
            raise UnexpectedSymbolError(capacity_name)


        list_var_names = self.MAP_CAPACITY[capacity_name]

        list_var_names = [var for var in list_var_names
                          if hasattr(self, var)]

        for var_name in list_var_names:

            if var_name in self.VARIABS_TIME:
                slot_objs = self.slots.values()
            elif var_name in self.VARIABS:
                slot_objs = [noneslot]

            cstr_name = 'cstr_%s_cap_%s'%(var_name, capacity_name)
            setattr(self, cstr_name, {})
            cstr_dict = getattr(self, cstr_name)

            for slot in slot_objs:

                base_name = '%s_%s_cap_%s_%s'%(self.name, var_name,
                                               capacity_name, str(slot.name))

                cstr = Constraint(base_name=base_name, slot=slot,
                                  var_name=str(getattr(self, var_name)[slot]))

                # define expression
                var = getattr(self, var_name)[slot]
                cap = getattr(self, capacity_name).symb

                # subtract retired capacity if applicable
                if (hasattr(self, capacity_name + '_ret')
                    # ... not for retired capacity constraint
                    and not capacity_name + '_ret' == var_name):
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

            cstr = Constraint(base_name=base_name, slot=slot,
                              var_name=str(getattr(self, variable)[slot]))

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


