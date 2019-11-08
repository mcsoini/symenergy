#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Part of symenergy. Copyright 2018 authors listed in AUTHORS.

Contains the Storage class describing assets which charge and discharge energy.


"""

import wrapt
import numpy as np
import itertools
from copy import copy
from hashlib import md5
from collections import Counter

import symenergy.core.asset as asset
from symenergy.core.constraint import Constraint
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter
from symenergy.auxiliary.constrcomb import CstrCombBase

from symenergy.core.slot import noneslot

from symenergy import _get_logger

logger = _get_logger(__name__)


class Storage(asset.Asset):
    r'''
    Parameters
    ----------
    name : str
        storage name
    eff : float
        storage round-trip efficiency
    slots_map : dict
        optional specification during which time slots storage can
        charge/discharge
    capacity : float
        Optional storage power capacity; this limits the charging and
        discharging power. Power is unconstrained if this
        argument is not set.
    energy_capacity : float
        Optional storage energy capacity; this limits the stored energy
        content. Power is unconstrained if this argument is not set.


    The time slot order follows from the order in which they are added to the
    model instance (:func:`symenergy.core.model.Model.add_slot`).


    The `slots_map` parameters defines during which time slots
    storage can charge or discharge. This allows for significant
    model simplifications. For example, the dictionary


    .. code-block:: python

         `{'chg': ['slot_1', 'slot_2'],
           'dch': ['slot_2']}


    allows for charging for slots `['slot_1', 'slot_2']` but limits
    discharging to `'slot_2'`.


    '''

    variabs = ['et']
    variabs_time = ['pchg', 'pdch', 'e']

    mutually_exclusive = {
        'Full storage can`t charge':
            (('e_cap_E', 'last', True), ('pos_pchg', 'this', False)),

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

        'All discharging zero -> each charging cannot be non-zero':
            (('pos_pdch', 'all', True), ('pos_pchg', 'this', False)),

        'All charging zero -> each energy cannot be non-zero':
            (('pos_pchg', 'all', True), ('pos_e', 'this', False)),

        'All discharging zero -> each energy cannot be non-zero':
            (('pos_pdch', 'all', True), ('pos_e', 'this', False)),
        }
    mutually_exclusive_no_blocks = {
        'Empty storage stays empty w/o charging_0':
            (('pos_e', 'anyprev', True), ('pos_pchg', 'lasts', True), ('pos_e', 'this', False)),
        'Empty storage stays empty w/o charging_1':
            (('pos_e', 'anyprev', True), ('pos_pchg', 'lasts', False), ('pos_e', 'this', True)),
#        'Empty storage stays empty w/o charging_2':  # << this combination is wrong and deletes valid solutions
#            (('pos_e', 'last', False), ('pos_pchg', 'this', True), ('pos_e', 'this', True)),
        'Full storage stays full w/o discharging_0':
            (('e_cap_E', 'anyprev', True), ('pos_pdch', 'lasts', True), ('e_cap_E', 'this', False)),
        'Full storage stays full w/o discharging_1':
            (('e_cap_E', 'anyprev', True), ('pos_pdch', 'lasts', False), ('e_cap_E', 'this', True)),
##        'Full storage stays full w/o discharging_2':  # << this combination is wrong and deletes valid solutions
##            (('e_cap_E', 'last', False), ('pos_pdch', 'this', True), ('e_cap_E', 'this', True)),

        'Empty storage can`t discharge':
            (('pos_e', 'last', True), ('pos_pdch', 'this', False)),

        'All energy zero -> each charging cannot be non-zero':
            (('pos_e', 'all', True), ('pos_pchg', 'this', False)),

        'All energy zero -> each discharging cannot be non-zero':
            (('pos_e', 'all', True), ('pos_pdch', 'this', False)),
        }

    def __init__(self, name, eff, slots_map=None, slots=None,
                 capacity=False, energy_capacity=False, energy_cost=1e-12,
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

        self._make_mutually_exclusive_dict()

        self.energy_cost = energy_cost

        for cd_slct in ['chg', 'dch']:
            slotsslct = self.slots_map[cd_slct]
            self._init_symbol_operation('p%s'%cd_slct, slotsslct)
        self._init_symbol_operation('e')

        for var in ['pchg', 'e', 'pdch']:
            self._init_cstr_positive(var)

        param_args = ('%s_%s'%('eff', self.name), noneslot, eff)
        self.eff = self.parameters.append(Parameter(*param_args))

        if self._slot_blocks:
            self._init_symbol_operation('et')
            self._init_cstr_slot_blocks_storage()

        lst_par = [('C', capacity), ('E', energy_capacity)]
        for param_name, param_val in lst_par:
            self._add_parameter(param_name, param_val, noneslot)

        self._init_cstr_storage()

        self._init_cost_component()

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


    def _make_mutually_exclusive_dict(self):
        '''
        Mutually exclusive dictionary depends on whether or not the model
        has slotblocks. The instance attribute is assembled from the class
        attributes accordingly.
        '''

        if not self._slot_blocks:
            self.mutually_exclusive.update(self.mutually_exclusive_no_blocks)


    def _init_prev_slot(self):
        '''
        Defines a dictionary with the previous slot for each time slot.
        '''

        def get_prev(slot_list):
            shifted_slots = np.roll(np.array(list(slot_list)), 1)
            return dict(zip(slot_list, shifted_slots))

        if not self._slot_blocks:
            dict_prev_slot = get_prev(self.slots.values())

        else:
            slots_by_block = {block: [slot for slot in self.slots.values()
                                      if slot.block == block]
                              for block in self._slot_blocks.values()}
            dict_prev_slot = dict(itertools.chain.from_iterable(
                                  [get_prev(slots).items()
                                   for slots in slots_by_block.values()]))

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
        This overwrites the symenergy.core.component method;
        For storage, CstrCombBase is initialized with `self._dict_prev_slot`
        instead of `self.slots`.
        '''

        list_col_names = []

#        mename, me = list(self.mutually_exclusive.items())[0]
        for mename, me in self.mutually_exclusive.items():

            list_cstrs = me
            slots_def = self._dict_prev_slot
            dict_cstrs = self.constraints.to_dict(dict_struct={'name_no_comp': {'slot': ''}})

            ccb = CstrCombBase(mename, list_cstrs, slots_def, dict_cstrs)
            list_col_names.append(ccb.gen_col_combs())

        list_col_names = list(itertools.chain.from_iterable(list_col_names))
        list_col_names = [cols for cols in list_col_names if cols]

        return list_col_names


    def expr_func_slot_blocks(self, slot):

            list_slots = [slot for slot in self.slots.values()
                          if slot.block
                             == list(self._slot_blocks.values())[0]]
            eff = self.eff.symb
            reps = self._slot_blocks[list_slots[0].block.name].rp.symb

            chg = sum(self.pchg[slot] * slot.w.symb
                      for slot in list_slots if slot in self.pchg)
            dch = sum(self.pdch[slot] * slot.w.symb
                      for slot in list_slots if slot in self.pdch)

            et = self.et[noneslot]

            return et - (chg * eff**(1/2) - dch / eff**(1/2)) * reps


    def _init_cstr_slot_blocks_storage(self):
        '''
        Defines the variable `et` as the first block's charging surplus.
        '''

        name = '%s_%s'%(self.name, 'def_et')


        cstr = Constraint(name, slot=noneslot,
                          expr_func=self.expr_func_slot_blocks,
                          is_equality_constraint=True, comp_name=self.name,
                          expr_args=(noneslot,))

        self.constraints.append(cstr)



    def expr_func_storage_noslots(self, slot, cd, sgn):
        expr = (sum(p * slot.w.symb
                for slot, p in getattr(self, 'p%s'%cd).items())
            * self.eff.symb**(sgn * 1/2)
            - self.e[noneslot])
        return expr


    def expr_func_storage_slots(self, slot):

        pchg = self.pchg[slot] if slot in self.pchg else 0
        pdch = self.pdch[slot] if slot in self.pdch else 0
        e = self.e[slot]
        e_prev = self.e[self._dict_prev_slot[slot]]

        slot_w = slot.w.symb
        expr = (e_prev
                + pchg * slot_w * self.eff.symb**(1/2)
                - pdch * slot_w * self.eff.symb**(-1/2)
                - e)

        def get_first (iblock):
            return [slot for slot in self.slots.values() if
                    slot.block == list(self._slot_blocks.values())[iblock]][0]

        if self._slot_blocks:

            # first slot of first block
            slot00 = get_first(0)
            slot10 = get_first(1)

            # first slot of first block --> subtract `et`
            # note: this needs to be consistent with the `cstr_def_et`
            # constraint
            if slot00 is slot:
                expr -= self.et[noneslot] / slot.block.rp.symb
            elif slot10 is slot:
                expr += self.et[noneslot] / slot.block.rp.symb

        return expr

    def _init_cstr_storage(self):
        ''' Initialize storage constraints. '''

        ###################
        # power to energy #
        if len(self.e) < 2:
            for cd, sgn in [('chg', +1), ('dch', -1)]:


                name = '%s_%s'%(self.name, 'pwrerg_%s'%cd)
                cstr = Constraint(name, slot=noneslot,
                                  expr_func=self.expr_func_storage_noslots,
                                  expr_args=(noneslot, cd, sgn),
                                  is_equality_constraint=True,
                                  comp_name=self.name)

                self.constraints.append(cstr)

        else:
            # e_t = e_t-1 + sqrt(eta) * pchg_t - 1 / sqrt(eta) * pdch_t

            self.cstr_pwrerg = {}

            if set(Counter([slot.block for slot
                            in self.slots.values()]).values()) == {1}:
                slot_list = [list(self.slots.values())[1]]
            else:
                slot_list = self.slots.values()


            for slot in slot_list:

                name = '%s_%s'%(self.name, 'pwrerg')

                cstr = Constraint(name, slot=slot,
                                  expr_func=self.expr_func_storage_slots,
                                  expr_args=(slot,),
                                  is_equality_constraint=True,
                                  comp_name=self.name)

                self.constraints.append(cstr)

    def _init_cost_component(self):
        '''
        Set constant and linear components of variable costs.
        '''

        self.cc = self.energy_cost * sum(self.e.values())


    def _get_component_hash_name(self):
        ''' Expand component hash.

        Storage has additional attributes which need to be included in the
        component hash value. These are:

        * `energy_cost`
        * `slots_map`
        '''

        hash_name_0 = super()._get_component_hash_name()
        hash_input = [str(self.slots_map),
                      '{:.20f}'.format(self.energy_cost),
                      str(self.energy_cost)]

        logger.debug('Generating storage hash.')

        return md5(str(hash_input + [hash_name_0]).encode('utf-8')).hexdigest()

