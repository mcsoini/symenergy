#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Storage class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import wrapt
import numpy as np
import itertools
from copy import copy
from hashlib import md5

import symenergy.core.asset as asset
from symenergy.core.constraint import Constraint
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter
from symenergy.auxiliary.constrcomb import CstrCombBase

from symenergy.core.slot import noneslot

from symenergy import _get_logger

logger = _get_logger(__name__)


class Storage(asset.Asset):
    '''
    Class describing storage assets which charge and discharge energy.

    The time slot order follows from the order in which they are added to the
    model instance (:func:`symenergy.core.model.Model.add_slot`).

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
    VARIABS = ['et']
    VARIABS_TIME = ['pchg', 'pdch', 'e']

    VARIABS_POSITIVE = ['p', 'e', 'C_ret', 'C_add', 'pchg', 'pdch']

    MUTUALLY_EXCLUSIVE = {
        'Empty storage stays empty w/o charging_0':
            (('pos_e', 'anyprev', True), ('pos_pchg', 'lasts', True), ('pos_e', 'this', False)),
        'Empty storage stays empty w/o charging_1':
#        'Empty storage stays empty w/o charging_2':  # << this combination is wrong and deletes valid solutions
#            (('pos_e', 'last', False), ('pos_pchg', 'this', True), ('pos_e', 'this', True)),

            (('pos_e', 'anyprev', True), ('pos_pchg', 'lasts', False), ('pos_e', 'this', True)),
        'Full storage stays full w/o discharging_0':
            (('e_cap_E', 'anyprev', True), ('pos_pdch', 'lasts', True), ('e_cap_E', 'this', False)),
        'Full storage stays full w/o discharging_1':
#        'Full storage stays full w/o discharging_2':  # << this combination is wrong and deletes valid solutions
#            (('e_cap_E', 'last', False), ('pos_pdch', 'this', True), ('e_cap_E', 'this', True)),

            (('e_cap_E', 'anyprev', True), ('pos_pdch', 'lasts', False), ('e_cap_E', 'this', True)),
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
            (('pos_pdch', 'all', True), ('pos_e', 'this', False)),

        'All discharging zero -> each charging cannot be non-zero':
            (('pos_pdch', 'all', True), ('pos_pchg', 'this', False)),

        'All energy zero -> each charging cannot be non-zero':
            (('pos_e', 'all', True), ('pos_pchg', 'this', False)),
        'All energy zero -> each discharging cannot be non-zero':
            (('pos_e', 'all', True), ('pos_pdch', 'this', False)),

        }

    def __init__(self, name, eff, slots_map=None, slots=None,
                 capacity=False, energy_capacity=False, energy_cost=1e-3,
                 _slot_blocks=None):

        super().__init__(name)
#        self.name = name
        self.slots = slots
        self._slot_blocks = _slot_blocks

        self.slots_map = (slots_map if slots_map
                          else {'chg': list(self.slots.keys()),
                                'dch': list(self.slots.keys())})

        self._update_class_attrs()
        self._init_prev_slot()

        self.energy_cost = energy_cost

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

        if self._slot_blocks:
            self.init_symbol_operation('et')
            self._init_cstr_slot_blocks_storage()

        if capacity:
            self.C = Parameter('C_%s'%self.name, noneslot, capacity)
            self._init_cstr_capacity('C')

        if energy_capacity:
            self.E = Parameter('E_%s'%self.name, noneslot, energy_capacity)
            self._init_cstr_capacity('E')

        self.init_cstr_storage()

        self.init_cost_component()

    @property
    def slots_map(self):
        return self._slots_map

    @slots_map.setter
    def slots_map(self, slots_map):
        '''
        Check that slots_map is a dictionary with keys *exactly* ['chg', 'dch']
        and values lists with subsets of slots.
        '''
        assert set(slots_map) == {'chg', 'dch'}, \
            'Invalid slots_map keys. Must be ["chg", "dch"]'

        assert (set(itertools.chain.from_iterable(slots_map.values()))
                    .issubset(set(self.slots))), \
            'Invalid slots_map values. Must be subsets of slots.'

        self._slots_map = slots_map


    def _init_prev_slot(self):
        '''
        Defines a dictionary with the previous slot for each time slot.
        Default case: All
        '''

        if not self._slot_blocks:

            shifted_slots = np.roll(np.array(list(self.slots.values())), 1)
            dict_prev_slot = dict(zip(self.slots.values(), shifted_slots))

        else:
            # 2x2 time slots
            list_slots = list(self.slots.values())
            dict_prev_slot = {list_slots[0]: list_slots[1],
                              list_slots[1]: list_slots[0],
                              list_slots[2]: list_slots[3],
                              list_slots[3]: list_slots[2],}

        self._dict_prev_slot = dict_prev_slot

    def _update_class_attrs(self):
        '''
        Update instances VARIABS class attribute in dependence on time slot.
        '''

        if (len(self.slots) == 1
            or len(self.slots_map['chg']) == 1
            or len(self.slots_map['dch']) == 1):

            logger.warning(('%s: Moving variable e from VARIABS_TIME '
                           'to VARIABS')%self.name)

            self.VARIABS_TIME = copy(self.VARIABS_TIME)  # copy class attr
            self.VARIABS_TIME.remove('e')
            self.VARIABS = copy(self.VARIABS) + ['e']


    def get_mutually_exclusive_cstrs(self):
        '''
        This overwrites the symenergy.core.component method; CstrCombBase
        is initialized with self._dict_prev_slot instead of self.slots.
        '''

        list_col_names = []

#        mename, me = list(self.MUTUALLY_EXCLUSIVE.items())[0]
        for mename, me in self.MUTUALLY_EXCLUSIVE.items():

            list_cstrs = me
            slots_def = self._dict_prev_slot
            dict_cstrs = self.get_constraints(by_slot=False, names=True)

            ccb = CstrCombBase(mename, list_cstrs, slots_def, dict_cstrs)

            list_col_names.append(ccb.gen_col_combs())

        list_col_names = list(itertools.chain.from_iterable(list_col_names))
        list_col_names = [cols for cols in list_col_names if cols]

        return list_col_names

    def _init_cstr_slot_blocks_storage(self):
        '''
        Defines the variable `et` as the first block's charging surplus.
        '''

        name = '%s_%s_%s'%(self.name, 'def_et', noneslot.name)
        cstr = Constraint(name, slot=noneslot, is_equality_constraint=True)

        list_slots = list(self.slots.values())[:2]
        eff = self.eff.symb
        reps = self._slot_blocks[list_slots[0].block].repetitions

        chg = sum(self.pchg[slot] * slot.w.symb
                  for slot in list_slots if slot in self.pchg)
        dch = sum(self.pdch[slot] * slot.w.symb
                  for slot in list_slots if slot in self.pdch)

        et = self.et[noneslot]

        cstr.expr = et - (chg * eff**(1/2) - dch / eff**(1/2)) * reps

        self.cstr_def_et = {noneslot: cstr}


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

                expr = (sum(p * slot.w.symb
                            for slot, p in getattr(self, 'p%s'%cd).items())
                        * self.eff.symb**(sgn * 1/2)
                        - self.e[noneslot])
                cstr_pwrerg.expr = expr

                setattr(self, 'cstr_pwrerg_%s'%cd, {noneslot: cstr_pwrerg})

        else:
            # e_t = e_t-1 + sqrt(eta) * pchg_t - 1 / sqrt(eta) * pdch_t

            self.cstr_pwrerg = {}

            for slot_name, slot in self.slots.items():

                name = '%s_%s_%s'%(self.name, 'pwrerg', slot.name)

                cstr = Constraint(name, slot=slot, is_equality_constraint=True)

                pchg = self.pchg[slot] if slot in self.pchg else 0
                pdch = self.pdch[slot] if slot in self.pdch else 0
                e = self.e[slot]
                e_prev = self.e[self._dict_prev_slot[slot]]

                slot_w = slot.w.symb
                expr = (e_prev
                        + pchg * slot_w * self.eff.symb**(1/2)
                        - pdch * slot_w * self.eff.symb**(-1/2)
                        - e)

                # first slot of first block --> subtract `et`
                # note: this needs to be consistent with the `cstr_def_et`
                # constraint
                if self._slot_blocks and list(self.slots.values())[0] is slot:
                    expr -= self.et[noneslot] / slot.repetitions
                elif self._slot_blocks and list(self.slots.values())[2] is slot:
                    expr += self.et[noneslot] / slot.repetitions

                cstr.expr = expr
                self.cstr_pwrerg[slot] = cstr

    def init_cost_component(self):
        '''
        Set constant and linear components of variable costs.
        '''

        self.cc = self.energy_cost * sum(self.e.values())


    def get_component_hash_name(self):
        ''' Expand component hash.

        Storage has additional attributes which need to be included in the
        component hash value. These are:

        * `energy_cost`
        * `slots_map`
        '''

        hash_name_0 = super().get_component_hash_name()
        hash_input = [str(self.slots_map),
                      '{:.20f}'.format(self.energy_cost),
                      str(self.energy_cost)]

        logger.debug('Generating storage hash.')

        return md5(str(hash_input + [hash_name_0]).encode('utf-8')).hexdigest()

