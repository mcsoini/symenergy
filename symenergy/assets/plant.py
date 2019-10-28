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
    Implements power plants with linear cost supply curve.

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
    VARIABS = ['C_ret']
    VARIABS_TIME = ['p']

    # mutually exclusive constraint combinations
    mutually_exclusive = {
# =============================================================================
# TODO: This needs to be fixed: C_ret defined for Noneslot
#         'Power plant retirement not simult. max end zero':
#             (('pos_C_ret', 'this', True), ('C_ret_cap_C', 'this', True)),
# =============================================================================
        'Power plant output not simult. max end zero':
            (('pos_p', 'this', True), ('p_cap_C', 'this', True))

        # C_ret max --> no output
        }

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

        super().__init__(name)
#        self.name = name

        self.slots = slots if slots else noneslot

        self._init_symbol_operation('p')

        self.vc0 = Parameter('vc0_%s'%self.name, noneslot, vc0)
        self.parameters.append(self.vc0)
        if vc1:
            self.vc1 = Parameter('vc1_%s'%self.name, noneslot, vc1)
            self.parameters.append(self.vc1)

        self._init_cstr_positive('p')

        if fcom:
            self.fcom = Parameter('fcom_%s'%self.name, noneslot, fcom)
            self.parameters.append(self.fcoms)

        if cap_ret:
            # needs to be initialized before _init_cstr_capacity('C')!
            self.init_symbol_operation('C_ret')
            self.init_cstr_positive('C_ret')

        if capacity:
            self.C = Parameter('C_%s'%self.name, noneslot, capacity)
            self.parameters.append(self.C)
            self._init_cstr_capacity('C')

        self._init_cost_component()


    def _init_cost_component(self):
        '''
        Set constant and linear components of variable costs.
        '''

        def get_vc(slot):
            return ((self.vc0.symb + self.vc1.symb * self.p[slot])
                    if hasattr(self, 'vc1')
                    else self.vc0.symb)

        self.vc = {slot: get_vc(slot) * slot.w.symb
                   * (slot.block.rp.symb if slot.block else 1)
                   for slot in self.slots.values()}

        self.cc = sum(sp.integrate(vc, self.p[slot])
                      for slot, vc in self.vc.items())

        if 'fcom' in self.__dict__:

            cc_fcom = self.C.symb * self.fcom.symb

            if hasattr(self, 'C_ret'):
                cc_fcom -= self.C_ret[noneslot] * self.fcom.symb

            self.cc += cc_fcom

