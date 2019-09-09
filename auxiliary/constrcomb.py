#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:01:01 2019

@author: user
"""

import wrapt
import itertools

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


    @none_if_invalid
    def gen_col_combs(self):
        '''
        For now, all constraints affect time variables (pdch, pchg, e). Because
        of this, we loop over the slots.
        '''

        list_slots = list(self.dict_prev_slot)

        dict_cstrs = self.get_cstr_objs()

        # mixes of None and other slots don't make sense
        found_slots = set([list(cstr.keys())[0] for cstr in dict_cstrs])
        flag_mix_none = (any(slot.name == 'None' for slot in found_slots) and
                         any(slot.name != 'None' for slot in found_slots))

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
                for c, dict_slot_cstr in zip(self.list_cstrs, dict_cstrs):
                    dict_slotcode_cols = {'all': [(dict_slot_cstr[slot].col, c[-1])
                                                   for slot in list_slots],
                                         'this': [(dict_slot_cstr[slot].col, c[-1])]}
                    list_col_names_slot.append(dict_slotcode_cols[c[1]])

                list_col_names.append(tuple(itertools.chain.from_iterable(list_col_names_slot)))

        else:
            # no need to loop over time slots
            list_col_names = []
            for c, dict_slot_cstr in zip(self.list_cstrs, dict_cstrs):

                dict_slotcode_cols = {'this': [(dict_slot_cstr[slot].col, c[-1]) for slot
                                               in self.dict_prev_slot.keys()],
                                      'last': [(dict_slot_cstr[slot].col, c[-1]) for slot
                                               in self.dict_prev_slot.values()]}

                list_col_names.append(dict_slotcode_cols[c[1]])

            # here, slots are done in parallel. need to zip to get cstr combs
            list_col_names = list(zip(*list_col_names))

        return list_col_names
