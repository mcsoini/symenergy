#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 21:01:01 2019

@author: user
"""

import wrapt
import itertools
import numpy as np
import pandas as pd
import operator
import functools

chain = itertools.chain.from_iterable

#from symenergy.core.slot import noneslot
from symenergy import _get_logger

logger = _get_logger(__name__)


def filter_constraint_combinations_gen(combs_gen, mutually_exclusive_cols,
                                   column_names):
    '''
    Low memory demand demand but slow when compared to pandas version...

    Call like:
    combs_gen = itertools.product(*bools)
    df_comb = filter_constraint_combinations_gen(combs_gen, mutually_exclusive_cols=mut_excl_cols,
                                                 column_names=constrs_cols_neq)

    '''

    cols = column_names

    # translate column names to column index
    mutually_exclusive_index = [[(cols.index(col[0]), col[1])
                                 for col in comb]
                                for comb in mutually_exclusive_cols]

    mutually_exclusive_index = [list(zip(*comb))
                                for comb in mutually_exclusive_index]

    if __name__ == '__main__':
        combs_gen = itertools.product(*bools)

    @functools.lru_cache(maxsize=1024, typed=True)
    def get_getter(list_):
        return operator.itemgetter(*list_)

    def check_comb(comb_inp, comb_check):
        return get_getter(comb_check[0])(comb_inp) == comb_check[1]

    df = pd.DataFrame([comb_inp for comb_inp in combs_gen
                       if not any(check_comb(comb_inp, comb_check)
                       for comb_check in mutually_exclusive_index)],
                       columns=cols)

    return df



def filter_constraint_combinations(df, mutually_exclusive_cols):

    ncombs = len(df)

    for cols in mutually_exclusive_cols:
        logger.info('Deleting constraint combination: %s'%str(cols))

        mask = pd.Series(True, index=df.index)

        for col, bool_act in cols:
            mask = mask & (df[col] == bool_act)

        df = df.loc[~mask]

        ndel = mask.sum()
        logger.info(('... total deleted: %s (%s), remaining: %s'
                     )%(ndel, '{:.1f}%'.format(ndel/ncombs*100),
                        len(df)))

    return df


class CstrCombBase():

    def __init__(self, mename, list_cstrs, slots_def, dict_cstrs):
        '''

        Parameters
        ----------
        mename : str
            name of mutually exclusive constraint combination; keys of the
            `MUTUALLY_EXCLUSIVE` class attribute dictionaries
        list_cstrs :
        slots_def : list or dict
            list of slots (used for power plants) or
            dictionary like {slot: previous slot} for storage
        constraints : dict
            dictionary of constraint objects of type
            `{'name_no_comp': {'slot': ''}}`

        '''

        self.mename = mename

        logger.info('Init CstrCombBase "%s"'%mename)

        flag_valid = \
            (isinstance(list_cstrs, tuple)
                and all(isinstance(list_cstr, tuple)
                        for list_cstr in list_cstrs))

        if not flag_valid:
            raise ValueError('list_cstrs needs to be tuples of tuples')

        self.list_cstrs = list_cstrs

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

            map_slots = np.isin(np.array(dict_slots[slottype]),
                                ar_slots_cstr)
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

    def expand_slots_all_this(self):
        '''
        Note: Returns [] if the storage energy variable e is defined only for
        the noneslot.

        Note: Ok for all the slot_block case.

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
        nlasts = nslots

        map_lasts = map_0[:, :nlasts]

        for n_this in range(nslots):  # loop over all slot indices (shift loop)

            dict_slots = {'all': ar_slots[map_lasts[n_this]],
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

        list_combs = []
        for slot_this, slot_last in self.dict_prev_slot.items():

            dict_slots = {'last': np.array([slot_last]),
                          'this': np.array([slot_this])}

            self._remove_nonexistent_slots(dict_slots, map_constr)

            # only proceed if constraint combination is valid given
            # remaining slots
            flag_valid = all(len(val) for val in dict_slots.values())

            if flag_valid:

                list_comb = []
                for comb, cstr in map_constr.items():

                    comb_slots_slct = dict_slots[comb[1]]
                    list_comb += tuple((cstr[slot].col, comb[-1])
                                       for slot in comb_slots_slct)

                list_combs.append(list_comb)

        return list_combs


    def expand_slots_this(self):

        dict_cstrs = self.get_cstr_objs()

        list_slots = list(self.dict_prev_slot)

        list_all_slots = []
        for slot in list_slots:
            list_combs = []
            for c, dict_cstr in list(zip(self.list_cstrs, dict_cstrs)):
                if slot in dict_cstr:
                    list_combs.append(tuple((dict_cstr[slot].col, c[-1])))

            if len(list_combs) == len(self.list_cstrs):
                # all relevant constraints exist
                list_all_slots.append(tuple(list_combs))

        return list_all_slots


    @none_if_invalid
    def gen_col_combs(self):

        list_code_rel_slot = set(cs[1] for cs in self.list_cstrs)

        logger.debug(list_code_rel_slot)

        if list_code_rel_slot == {'anyprev', 'lasts', 'this'}:

            list_col_names = self.expand_slots_anyprev_lasts_this()

        elif list_code_rel_slot == {'all', 'this'}:

            list_col_names = self.expand_slots_all_this()

        elif list_code_rel_slot == {'last', 'this'}:

            list_col_names = self.expand_slots_last_this()

        elif list_code_rel_slot == {'this'}:

            list_col_names = self.expand_slots_this()

        else:
            raise ValueError('Not implemented: list_code_rel_slot='
                             '%s'%list_code_rel_slot)

        kw_format = dict(ncols=len(list_col_names), cols=str(list_col_names))
        logger.info(('... expanded to {ncols} column combinations: {cols}'
                     ).format(**kw_format))

        return list_col_names
