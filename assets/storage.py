#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Storage class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""



import symenergy.core.asset as asset
from symenergy.core.constraint import Constraint
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter

class Storage(asset.Asset):
    '''
    For simplicity, storage is modelled with fixed time slots for
    charging and discharging. Like this the time slots don't require an
    order.
    '''

    PARAMS = ['eff', 'C', 'E']
    PARAMS_TIME = []
    VARIABS = ['e']
    VARIABS_TIME = ['p']

    VARIABS_POSITIVE = ['p', 'e', 'C_ret', 'C_add']

    MUTUALLY_EXCLUSIVE = [('pos_C_ret', 'C_ret_cap_C'),
                          ('pos_p', 'p_cap_C'),
                          ('pos_e', 'e_cap_E')]

    def __init__(self, name, eff, slots_map=None, slots=None,
                 capacity=False, energy_capacity=False):

        super().__init__()

        self.slots = slots if slots else {'day': Slot('day', 0, 0),
                                          'night': Slot('night', 0, 0)}
        self.slots_map = (slots_map if slots_map
                          else {list(self.slots.keys())[0]: 'chg',
                                list(self.slots.keys())[-1]: 'dch'})

        self.name = name

        # only one power for each slot, ... charging or discharging is
        # determined in the constraints depending on the specification in
        # the slots_map
        self.init_symbol_operation('p')
        self.init_symbol_operation('e')

        self.init_cstr_positive('p')
        self.init_cstr_positive('e')

        self.eff = Parameter('%s_%s'%('eff', self.name), noneslot, eff)

        if capacity:
            self.C = Parameter('C_%s'%self.name, noneslot, capacity)
            self.init_cstr_capacity('C')

        self.init_is_capacity_constrained('C', 'p')

        if energy_capacity:
            self.E = Parameter('E_%s'%self.name, noneslot, energy_capacity)
            self.init_cstr_capacity('E')

        self.init_is_capacity_constrained('E', 'e')

        self.init_is_positive()

        self.init_cstr_storage()


    def get_chgdch(self, dr):
        '''
        Return p attribute filtered by charging discharging.

        Parameter:
            * dr -- string, one of 'chg'/'dch'
        '''

        slots_rev = {vv: kk for kk, vv in self.slots.items()}

        return {slot: var for slot, var in self.p.items()
                if (slots_rev[slot] in self.slots_map.keys())
                and (self.slots_map[slots_rev[slot]] == dr)}

    def init_cstr_storage(self):
        '''
        Initialize storage constraints.
        '''

        name = '%s_%s_%s'%(self.name, 'store', noneslot.name)
        cstr_store = Constraint(name, slot=noneslot,
                                is_equality_constraint=True)
        name = '%s_%s_%s'%(self.name, 'pwrerg', noneslot.name)
        cstr_pwrerg = Constraint(name, slot=noneslot,
                                 is_equality_constraint=True)

        expr = (sum(p for _, p in self.get_chgdch('chg').items())
                    * self.eff.symb
                    - sum(p for _, p in self.get_chgdch('dch').items()))
        cstr_store.expr = expr * cstr_store.mlt
        self.cstr_store = {noneslot: cstr_store}

        # power to energy
        expr = (sum(p for _, p in self.get_chgdch('chg').items())
                * self.eff.symb**(1/2)
                - self.e[noneslot])
        cstr_pwrerg.expr = expr * cstr_pwrerg.mlt

        self.cstr_pwrerg = {noneslot: cstr_pwrerg}


