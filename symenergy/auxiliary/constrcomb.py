#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:01:01 2019

@author: user
"""

import wrapt
import itertools
import numpy as np

chain = itertools.chain.from_iterable

#from symenergy.core.slot import noneslot
from symenergy import _get_logger

logger = _get_logger(__name__)

class CstrCombBase():

    def __init__(self, mename, list_cstrs, slots_def, dict_cstrs):
        '''

        Parameters
        ----------
        mename -- str
            Name of mutually exclusive constraint combination. Probably the
            keys of the `MUTUALLY_EXCLUSIVE` class attribute dictionaries
        list_cstrs --
        slots_def -- list or dict
            list of slots (used for power plants) or
            dictionary like {slot: previous slot} for storage
        dict_cstrs --
        '''

        self.mename = mename

        logger.info('Init CstrCombBase "%s"'%mename)

        flag_valid = \
            (isinstance(list_cstrs, tuple)
                and all(isinstance(list_cstr, tuple)
                        for list_cstr in list_cstrs))

        if not flag_valid:
            raise ValueError('list_cstrs needs to be tuples of tuples')

        self.list_cstrs = [('cstr_%s'%name, relslot, bool_act)
                           for name, relslot, bool_act in list_cstrs]

        if isinstance(slots_def, list):
            self.dict_prev_slot = {slot: slot for slot in slots_def}
        else:
            self.dict_prev_slot = slots_def

        self.dict_cstrs = dict_cstrs

    def get_flat(self, cstr_only=False):

        cstrs_flat = self.list_cstrs

        return cstrs_flat if not cstr_only else [c[0] for c in cstrs_flat]


    def get_cstr_objs(self):

        return [self.dict_cstrs[cstr] for cstr in self.get_flat(True)]

    def all_cstrs_exist(self):
        '''
        Instance is considered an invalid constraint combination if
        this returns False.
        '''

        return tuple(cstr for cstr in self.get_flat(True)
                if cstr not in self.dict_cstrs)

    @wrapt.decorator
    def none_if_invalid(wrapped, self, args, kwargs):

        missing_cstrs = self.all_cstrs_exist()
        if missing_cstrs:
            logger.warning(('Aborting %s: missing constraints %s'
                            )%(wrapped.__name__, missing_cstrs))
            return []
        else:
            return wrapped(*args, **kwargs)

    @staticmethod
    def _remove_nonexistent_slots(dict_slots, map_constr):
        '''
        Remove nonexistent slots given an incomplete storage slots_map.

        For example, pchg might not be defined for all time slots of
        the model.
        '''

        # generate dict_slots_net based
        for comb, cstr in map_constr.items():

            slottype = comb[1]
            ar_slots_cstr = np.array(list(cstr.keys()))

            map_slots = np.isin(dict_slots[slottype], ar_slots_cstr)
            dict_slots[slottype] = dict_slots[slottype][map_slots]


    def expand_slots_anyprev_lasts_this(self):
        '''
        Note: Returns [] if the storage energy variable e is defined only for
        the noneslot.
        TODO: Implement separate case which covers e being defined for the
        noneslot only.
        '''

        dict_cstrs = self.get_cstr_objs()
        # dictionary constraint spec from const comb --> constraint dict by slot
        map_constr = dict(zip(self.list_cstrs, dict_cstrs))

        ar_slots = np.array(list(self.dict_prev_slot))
        nslots = len(ar_slots)

        # setup basic slot-slot index map
        rngs = [range(nslots)] * 2
        map_0 = np.mod(np.diff(np.array(np.meshgrid(*rngs)), axis=0)[0], nslots)

        list_combs = []
        for nlasts in range(1, nslots):  # from 1 to nslots - 1, inclusive (size loop)

            map_lasts = map_0[:, :nlasts]
            map_anypr = map_0[:, [nlasts]]

            for n_this in range(nslots):  # loop over all slot indices (shift loop)

                dict_slots = {'lasts': ar_slots[map_lasts[n_this]],
                              'anyprev': ar_slots[map_anypr[n_this]],
                              'this': np.array([ar_slots[n_this]])}

                self._remove_nonexistent_slots(dict_slots, map_constr)

                # only proceed if constraint combination is valid given
                # remaining slots
                flag_valid = all(val.size > 0 for val in dict_slots.values())

                if flag_valid:

                    list_comb = []
                    for comb, cstr in map_constr.items():

                        comb_slots_slct = dict_slots[comb[1]]
                        list_comb += tuple((cstr[slot].col, comb[-1]) for slot in comb_slots_slct)

                    list_combs.append(list_comb)

        return list_combs

    def expand_slots_last_this(self):

        dict_cstrs = self.get_cstr_objs()
        # dictionary constraint spec from const comb --> constraint dict by slot
        map_constr = dict(zip(self.list_cstrs, dict_cstrs))

        ar_slots = np.array(list(self.dict_prev_slot))
        nslots = len(ar_slots)

        # setup basic slot-slot index map
        rng = np.expand_dims(np.arange(nslots), 1)
        map_last = np.concatenate([rng, np.roll(rng, 1)], axis=1)

        list_combs = []
        for n_this in range(nslots):  # loop over all slot indices (shift loop)

            dict_slots = {'last': np.array([ar_slots[map_last[n_this, -1]]]),
                          'this': np.array([ar_slots[n_this]])}

            self._remove_nonexistent_slots(dict_slots, map_constr)

            # only proceed if constraint combination is valid given
            # remaining slots
            flag_valid = all(val.size > 0 for val in dict_slots.values())

            if flag_valid:

                list_comb = []
                for comb, cstr in map_constr.items():

                    comb_slots_slct = dict_slots[comb[1]]
                    list_comb += tuple((cstr[slot].col, comb[-1]) for slot in comb_slots_slct)

                list_combs.append(list_comb)

        return list_combs



    def expand_slots_this(self):

        dict_cstrs = self.get_cstr_objs()

        # no need to loop over time slots
        list_col_names = []
        for c, dict_slot_cstr in zip(self.list_cstrs, dict_cstrs):

            dict_code_slots = dict(zip(['this', 'last'],
                                       zip(*self.dict_prev_slot.items())))
            dict_code_slots = {kk: [(dict_slot_cstr[slot].col, c[-1])
                                    for slot in slots
                                    if slot in dict_slot_cstr]
                               for kk, slots in dict_code_slots.items()}

            list_col_names.append(dict_code_slots[c[1]])

        return list(zip(*list_col_names))

    @none_if_invalid
    def gen_col_combs(self):

        list_code_rel_slot = set(cs[1] for cs in self.list_cstrs)

        if list_code_rel_slot == {'anyprev', 'lasts', 'this'}:

            return self.expand_slots_anyprev_lasts_this()

        elif list_code_rel_slot == {'last', 'this'}:

            return self.expand_slots_last_this()

        elif list_code_rel_slot == {'this'}:

            return self.expand_slots_this()

        else:
            raise ValueError('Not implemented: list_code_rel_slot='
                             '%s'%list_code_rel_slot)


    @none_if_invalid
    def gen_col_combs_obsolete(self):
        '''
        For now, all constraints affect time variables (pdch, pchg, e). Because
        of this, we loop over the slots.
        '''

        list_slots = list(self.dict_prev_slot)
        dict_cstrs = self.get_cstr_objs()

        # mixes of None and other slots don't make sense
        set_slot_names = set(slot.name for slot in chain(dict_cstrs))
        flag_mix_none = 'None' in set_slot_names and len(set_slot_names) > 1

        if flag_mix_none:
            logger.warning('Found invalid combination of noneslot and other '
                           'slot. Constraint combination not applicable.')
            return []


        list_code_rel_slot = [cs[1] for cs in self.list_cstrs]

        # only considering combinations of 'all' and 'this' for now
        assert (not 'all' in list_code_rel_slot) or (2 == len(list_code_rel_slot))
        assert not(('all' in list_code_rel_slot) and (not 'this' in list_code_rel_slot))  # "all" only combined with 'this'
        # relative slot codes are these three:
        assert {'all', 'this', 'last'}.issuperset(set(list_code_rel_slot))

        # get all relevant combinations of time slots
        list_col_names = []
        if 'all' in list_code_rel_slot:
            for slot in list_slots:
                list_col_names_slot = []
                c, dict_slot_cstr = list(zip(self.list_cstrs, dict_cstrs))[0]
                for c, dict_slot_cstr in zip(self.list_cstrs, dict_cstrs):
                    dict_slotcode_cols = {'all': [(dict_slot_cstr[slot].col, c[-1])
                                                   for slot in list_slots
                                                   if slot in dict_slot_cstr],
                                         'this': [(dict_slot_cstr[slot].col, c[-1])]
                                                  if slot in dict_slot_cstr else []}
                    list_col_names_slot.append(dict_slotcode_cols[c[1]])

                list_col_names.append(tuple(chain(list_col_names_slot)))

            list_col_names = [cstr_comb for cstr_comb in list_col_names
                              if len(cstr_comb) >= len(self.list_cstrs)]
            return list_col_names

        else:
            # no need to loop over time slots
            list_col_names = []
            for c, dict_slot_cstr in zip(self.list_cstrs, dict_cstrs):

                dict_code_slots = dict(zip(['this', 'last'],
                                           zip(*self.dict_prev_slot.items())))
                dict_code_slots = {kk: [(dict_slot_cstr[slot].col, c[-1])
                                        for slot in slots
                                        if slot in dict_slot_cstr]
                                   for kk, slots in dict_code_slots.items()}

                list_col_names.append(dict_code_slots[c[1]])

            # if not all slot lists have the same length, the whole
            # constraint combination is invalid. This happens e.g. if the
            # slots map is used to define charging/discharging slots
            if len(set(len(cstr) for cstr in list_col_names)) != 1:
                return list_col_names
            else:
                # here, slots are done in parallel. need to zip to get cstr combs
                return list(zip(*list_col_names))





