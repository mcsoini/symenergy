#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the symenergy Component class.


Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import itertools
import pandas as pd
from symenergy.auxiliary.constrcomb import CstrCombBase
from symenergy import _get_logger

logger = _get_logger(__name__)



class Component():
    '''
    Make sure that all children implement PARAMS, VARIABS AND MULTIPS
    '''
    def __init__(self, name):

        self.name = name

    def get_params_dict(self, attr=tuple()):

        attr = tuple(attr) if isinstance(attr, str) else attr

        param_objs = \
                [getattr(self, param_name)
                 for param_name in self.PARAMS_TIME + self.PARAMS
                 if hasattr(self, param_name)]

        if len(attr) == 1:
            return [getattr(par, attr[0])
                    for par in param_objs]
        elif len(attr) == 2:
            return {getattr(par, attr[0]): getattr(par, attr[1])
                    for par in param_objs}
        else:
            return param_objs

    def get_constraint_combinations(self):
        '''
        Gathers all non-equal component constraints,
        calculates cross-product of all
        (binding, non-binding) combinations and instantiates dataframe.
        '''

        constrs_neq = [cstr for cstr in self.get_constraints() if not
                       cstr.is_equality_constraint]
        constrs_cols_neq = [cstr.col for cstr in constrs_neq]

        mutually_exclusive_cols = self.get_mutually_exclusive_cstrs()

        ncombs = 2**len(constrs_neq)

        logger.warning(('Component %s: Generating df_comb with length %d'
                        )%(self.name, ncombs))

        bools = [[False, True] for cc in constrs_neq]
        df_comb = pd.DataFrame(itertools.product(*bools),
                               columns=constrs_cols_neq, dtype=bool)

        for cols in mutually_exclusive_cols:
            logger.info('Deleting constraint combination: %s'%str(cols))

            mask = pd.Series(True, index=df_comb.index)

            for col, bool_act in cols:
                mask = mask & (df_comb[col] == bool_act)

            df_comb = df_comb.loc[~mask]

            ndel = mask.sum()
            logger.info(('... total deleted: %s (%s), remaining: %s'
                         )%(ndel, '{:.1f}%'.format(ndel/ncombs*100),
                            len(df_comb)))

        self._df_comb = df_comb

        return df_comb


    def get_is_capacity_constrained(self):
        '''
        Returns a tuple of all variables defined by the MAP_CAPACITY dict.

        Only include if the capacity is defined.
        '''

        return tuple(var
                     for cap_name, var_names in self.MAP_CAPACITY.items()
                     for var_name in var_names
                     if hasattr(self, var_name) and hasattr(self, cap_name)
                     for slot, var in getattr(self, var_name).items())

    def get_is_positive(self):
        '''
        Returns a tuple of all variables defined by the VARIABS_POSITIVE list.
        '''

        return tuple(var
                     for var_name in self.VARIABS_POSITIVE
                     if hasattr(self, var_name)
                     for slot, var in getattr(self, var_name).items())


    def get_constraints(self, by_slot=True, names=False):
        '''
        Returns
        -------
        list of constraint objects

        '''

        constrs = {} if names else []

        attrs = [(key, attr) for key, attr in vars(self).items() if
                 key.startswith('cstr_')]  # naming convention of constraint attrs

        for key, attr in attrs:
            if by_slot:
                for slot, cstr in attr.items():
                    if names:
                        constrs[key] = cstr
                    else:
                        constrs.append(cstr)
            else:
                if names:
                    constrs[key] = attr
                else:
                    constrs.append(attr)

        return constrs



    def get_mutually_exclusive_cstrs(self):
        '''
        Time dependent mutually inclusive constraints.
        '''

        list_col_names = []
        for mename, me in self.MUTUALLY_EXCLUSIVE.items():

            ccb = CstrCombBase(mename, me, list(self.slots.values()),
                               self.get_constraints(by_slot=False, names=True))

            list_col_names.append(ccb.gen_col_combs())


        return list(itertools.chain.from_iterable(list_col_names))


    def get_mutually_inclusive_cstrs(self):
        '''
        Implemented in child classes
        '''

        return []


    def get_variabs(self):
        '''
        Collect all variables of this component.

        Return values:
            list of all variable symbols
        '''

        return [vv
                for var in self.VARIABS + self.VARIABS_TIME
                if hasattr(self, var)
                for vv in getattr(self, var).values()]


