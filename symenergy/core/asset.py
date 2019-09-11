#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Asset class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""


import sympy as sp

import symenergy.core.component as component
from symenergy.core.constraint import Constraint
from symenergy.core.parameter import Parameter
from symenergy.core.slot import noneslot

from symenergy import _get_logger

logger = _get_logger(__name__)

class UnexpectedSymbolError(Exception):

    def __init__(self, res):

        message = ('Method called with unexpected '
                   'variable/parameter %s'%(res))

        super().__init__(message)

class Asset(component.Component):
    '''Mixin class containing shared methods of plants and storage'''

    # capacity class map
    MAP_CAPACITY = {'C': ['p', 'pchg', 'pdch', 'C_ret'],  # all power and retired capacity
                    'E': ['e']}  # storage energy capacity

    def __init__(self, name):

        super(Asset, self).__init__(name)
#        self.name = name

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
        list_var_names = [var for var in list_var_names if hasattr(self, var)]

        for var_name in list_var_names:  # loop over constrained variables

            slot_objs = (self.slots.values() if self.get_flag_timedep(var_name)
                         else [noneslot])

            cstr_name = 'cstr_%s_cap_%s'%(var_name, capacity_name)
            setattr(self, cstr_name, {})  # initialize instance dict attribute
            cstr_dict = getattr(self, cstr_name)

            var_attr = getattr(self, var_name)

            for slot in set(slot_objs) & set(var_attr):
                base_name = '%s_%s_cap_%s_%s'%(self.name, var_name,
                                               capacity_name, str(slot.name))

                cstr = Constraint(base_name=base_name, slot=slot,
                                  var_name=str(var_attr[slot]))

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

        slot_objs = (self.slots.values() if self.get_flag_timedep(variable)
                     else [noneslot])

        setattr(self, 'cstr_pos_%s'%variable, dict())
        cstr_dict = getattr(self, 'cstr_pos_%s'%variable)

        var_attr = getattr(self, variable)

        for slot in set(slot_objs) & set(var_attr):

            base_name = '%s_pos_%s_%s'%(self.name, variable, str(slot.name))

            cstr = Constraint(base_name=base_name, slot=slot,
                              var_name=str(var_attr[slot]))

            var = var_attr[slot]
            cstr.expr = cstr.mlt * var

            cstr_dict[slot] = cstr


    def get_flag_timedep(self, variable):
        '''
        TODO: The first case should depend on the chg/dch slots.
        '''


        if variable in set(self.VARIABS) & set(self.VARIABS_TIME):
            # the variable is defined for all time slots only if there are
            # two or more time slots (used for stored energy)

            flag_timedep = len(self.slots) >= 2

        elif variable in set(self.VARIABS) | set(self.VARIABS_TIME):
            flag_timedep = variable in self.VARIABS_TIME

        else:
            raise UnexpectedSymbolError(variable)

        logger.info('Variable %s has time dependence %s'%(variable,
                                                          flag_timedep))

        return flag_timedep


    def init_symbol_operation(self, variable, slotsslct=None):
        '''
        Sets operational variables, i.e. power (generation,
        charging, discharging) and stored energy.

        Parameters
        ----------
        variable: str
            collective variable name to be added
        slots: list of strings
            list of time slot names for which the variable is to be defined;
            defaults to all time slots

        '''

        flag_timedep = self.get_flag_timedep(variable)

        if not flag_timedep:
            symb = sp.symbols('%s_%s_%s'%(self.name, variable, str(None)))
            setattr(self, variable, {noneslot: symb})

        else:
            slots = ({slot_name: slot for slot_name, slot in self.slots.items()
                      if slot_name in slotsslct}
                     if slotsslct else self.slots)
            var_obj = {slot: sp.symbols('%s_%s_%s'%(self.name, variable,
                                        str(slot.name)))
                       for slot in slots.values()}
            setattr(self, variable, var_obj)

    def init_symbols_costs(self):
        ''' Overridden by children, if applicable. '''

    def _subs_cost(self, symb, *args, **kwargs):
        ''' Overridden by children, if applicable. '''

        return symb

    def __repr__(self):

        return '%s %s'%(self.__class__, str(self.name))

