#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Plant class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import sympy as sp

import symenergy.core.asset as asset
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter

class Plant(asset.Asset):
    '''
    Does know about slots.
    '''
    '''
    All plants have:
        * symbol power production p
        * symbol costs vc_0 and vc_1
        * cost component (vc_0 + 0.5 * vc_1 * p) * p
        * multiplier > 0
    Some plants have:
        * symbol capacity C
        * value capacity
        * multiplier power p <= capacity C
    '''
    PARAMS = ['vc0', 'vc1', 'C', 'fcom']
    PARAMS_TIME = []
    VARIABS = ['C_ret']
    VARIABS_TIME = ['p']

    VARIABS_POSITIVE = ['p', 'C_ret', 'C_add']

    # mutually exclusive constraint combinations
    MUTUALLY_EXCLUSIVE = [('pos_C_ret', 'C_ret_cap_C'),
                          ('pos_p', 'p_cap_C')]

    def __init__(self, name, vc0, vc1=None,
                 fcom=None, slots=None, capacity=False, cap_ret=False):

        '''
        Params:
            * name --
            * vc0 --
            * vc1 --
            * fcom -- float, O&M fixed cost
            * slots -- iterable of time slot names
            * capacity --
            * cap_ret -- boolean, capacity can be retired True/False

        TODO: Make vc1 optional.
        '''
        super().__init__()

        self.slots = slots if slots else {'0': Slot('0', 0, 0)}

        self.name = name

        self.init_symbol_operation('p')

        self.vc0 = Parameter('vc0_%s'%self.name, noneslot, vc0)
        if vc1:
            self.vc1 = Parameter('vc1_%s'%self.name, noneslot, vc1)

        self.init_cstr_positive('p')

        if fcom:
            self.fcom = Parameter('fcom_%s'%self.name, noneslot, fcom)

        if cap_ret:
            # needs to be initialized before init_cstr_capacity('C')!
            self.init_symbol_operation('C_ret')
            self.init_cstr_positive('C_ret')


        if capacity:

            self.C = Parameter('C_%s'%self.name, noneslot, capacity)
            self.init_cstr_capacity('C')


        self.init_cost_component()

        self.init_is_capacity_constrained('C', 'p')
        self.init_is_positive()

    def init_cost_component(self):
        '''
        Set constant and linear components of variable costs.
        '''

        if hasattr(self, 'vc1'):
            self.vc = {slot: self.vc0.symb + self.vc1.symb * self.p[slot]
                       for slot in self.slots.values()}
        else:
            self.vc = {slot: self.vc0.symb for slot in self.slots.values()}

        self.cc = sum(sp.integrate(vc, self.p[slot])
                         for slot, vc in self.vc.items())

        if 'fcom' in self.__dict__:

            cc_fcom = self.C.symb * self.fcom.symb

            if hasattr(self, 'C_ret'):
                cc_fcom -= self.C_ret[noneslot] * self.fcom.symb

            self.cc += cc_fcom

