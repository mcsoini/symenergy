#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Storage class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import numpy as np

import symenergy.core.asset as asset
from symenergy.core.constraint import Constraint
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter

from symenergy import _get_logger

logger = _get_logger(__name__)


class Storage(asset.Asset):
    '''
    For simplicity, storage is modelled with fixed time slots for
    charging and discharging. Like this the time slots don't require an
    order.

    Parameters
    ----------
    name : str
        name of the new component
    eff : float
        round trip efficiency
    slots_map : dict
        dictionary `{'chg': ["slot_1", "slot_2", ...],
                     'dch': ["slot_2", ...]}`
    slots : dict
        model instance slot dictionary passed to the initializer
    capacity : float
        storage power capacity
    energy_capacity : float
        storage energy capacity

    '''

    PARAMS = ['eff', 'C', 'E']
    PARAMS_TIME = []
    VARIABS = ['e']
    VARIABS_TIME = ['pchg', 'pdch', 'e']

    VARIABS_POSITIVE = ['p', 'e', 'C_ret', 'C_add', 'pchg', 'pdch']

    MUTUALLY_EXCLUSIVE = {
        'Empty storage stays empty w/o charging_0':
            (('pos_e', 'last', True), ('pos_pchg', 'this', True), ('pos_e', 'this', False)),
        'Empty storage stays empty w/o charging_1':
            (('pos_e', 'last', True), ('pos_pchg', 'this', False), ('pos_e', 'this', True)),
#        'Empty storage stays empty w/o charging_2':  # << this combination is wrong and deletes valid solutions
#            (('pos_e', 'last', False), ('pos_pchg', 'this', True), ('pos_e', 'this', True)),

        'Full storage stays full w/o discharging_0':
            (('e_cap_E', 'last', True), ('pos_pdch', 'this', True), ('e_cap_E', 'this', False)),
        'Full storage stays full w/o discharging_1':
            (('e_cap_E', 'last', True), ('pos_pdch', 'this', False), ('e_cap_E', 'this', True)),
#        'Full storage stays full w/o discharging_2':  # << this combination is wrong and deletes valid solutions
#            (('e_cap_E', 'last', False), ('pos_pdch', 'this', True), ('e_cap_E', 'this', True)),

        'Full storage can`t charge':
            (('e_cap_E', 'last', True), ('pos_pchg', 'this', False)),
        'Empty storage can`t discharge':
            (('pos_e', 'last', True), ('pos_pdch', 'this', False)),

        'No simultaneous non-zero charging and non-zero discharging':
            (('pos_pchg', 'this', False), ('pos_pdch', 'this', False)),
        'No simultaneous full-power charging and full-power discharging':
            (('pchg_cap_C', 'this', True), ('pdch_cap_C', 'this', True)),

        'Storage energy not simult. full and empty':
            (('pos_e', 'this', True), ('e_cap_E', 'this', True)),
        'Storage charging not simult. max end zero':
            (('pos_pchg', 'this', True), ('pchg_cap_C', 'this', True)),
        'Storage discharging not simult. max end zero':
            (('pos_pdch', 'this', True), ('pdch_cap_C', 'this', True)),

        'All charging zero -> each discharging cannot be non-zero':
            (('pos_pchg', 'all', True), ('pos_pdch', 'this', False)),
        'All charging zero -> each energy cannot be non-zero':
            (('pos_pchg', 'all', True), ('pos_e', 'this', False)),
        'All discharging zero -> each energy cannot be non-zero':
            (('pos_dchg', 'all', True), ('pos_e', 'this', False)),
        'All discharging zero -> each charging cannot be non-zero':
            (('pos_dchg', 'all', True), ('pos_pchg', 'this', False)),
        'All energy zero -> each charging cannot be non-zero':
            (('pos_e', 'all', True), ('pos_pchg', 'this', False)),
        'All energy zero -> each discharging cannot be non-zero':
            (('pos_e', 'all', True), ('pos_pdch', 'this', False)),

        }

    def __init__(self, name, eff, slots_map=None, slots=None,
                 capacity=False, energy_capacity=False):

        super().__init__()

        self.slots = slots

        self.slots_map = (slots_map if slots_map
                          else {'chg': list(self.slots.keys()),
                                'dch': list(self.slots.keys())})

        self.name = name

        # only one power for each slot, ... charging or discharging is
        # determined in the constraints depending on the specification in
        # the slots_map

        for cd_slct in ['chg', 'dch']:
            slotsslct = self.slots_map[cd_slct]
            self.init_symbol_operation('p%s'%cd_slct, slotsslct)
        self.init_symbol_operation('e')

        self.init_cstr_positive('pchg')
        self.init_cstr_positive('pdch')
        self.init_cstr_positive('e')

        self.eff = Parameter('%s_%s'%('eff', self.name), noneslot, eff)

        if capacity:
            self.C = Parameter('C_%s'%self.name, noneslot, capacity)
            self.init_cstr_capacity('C')

        if energy_capacity:
            self.E = Parameter('E_%s'%self.name, noneslot, energy_capacity)
            self.init_cstr_capacity('E')

        self.init_cstr_storage()

    def init_cstr_storage(self):
        '''
        Initialize storage constraints.
        '''

        ###################
        # power to energy #
        if len(self.e) < 2:
            for cd, sgn in [('chg', +1), ('dch', -1)]:
                name = '%s_%s_%s'%(self.name, 'pwrerg_%s'%cd, noneslot.name)
                cstr_pwrerg = Constraint(name, slot=noneslot,
                                         is_equality_constraint=True)

                expr = (sgn * sum(p * (slot.weight if slot.weight else 1)
                            for slot, p in self.pchg.items())
                        * self.eff.symb**(sgn * 1/2)
                        - self.e[noneslot])
                cstr_pwrerg.expr = expr * cstr_pwrerg.mlt

                setattr(self, 'cstr_pwrerg_%s'%cd, {noneslot: cstr_pwrerg})

        else:
            # e_t = e_t-1 + sqrt(eta) * pchg_t - 1 / sqrt(eta) * pdch_t

            # select time slot
            shifted_slots = np.roll(np.array(list(self.slots.values())), 1)
            dict_prev_slot = dict(zip(self.slots.values(), shifted_slots))

            self.cstr_pwrerg = {}

            for slot_name, slot in self.slots.items():

                name = '%s_%s_%s'%(self.name, 'pwrerg', slot.name)

                cstr = Constraint(name, slot=slot, is_equality_constraint=True)

                pchg = self.pchg[slot] if slot in self.pchg else 0
                pdch = self.pdch[slot] if slot in self.pdch else 0
                e = self.e[slot]
                e_prev = self.e[dict_prev_slot[slot]]

                slot_w = (slot.weight if slot.weight else 1)
                expr = (e_prev
                        + pchg * slot_w * self.eff.symb**(1/2)
                        - pdch * slot_w * self.eff.symb**(-1/2)
                        - e)

                cstr.expr = expr * cstr.mlt

                self.cstr_pwrerg[slot] = cstr




