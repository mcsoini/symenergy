#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the symenergy Component class.


Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import itertools
from hashlib import md5
import pandas as pd
from symenergy.auxiliary.constrcomb import CstrCombBase
from symenergy.auxiliary.constrcomb import filter_constraint_combinations
from symenergy import _get_logger

logger = _get_logger(__name__)

class Component():
    '''
    Base class for components.

    This serves as parent class for

    - time slots (:class:`symenergy.core.slot.Slot`)
    - assets (:class:`symenergy.core.asset.Asset`)

    It is not instantiated directly.

    '''

    def __init__(self, name):

        self.name = name
        self._check_attributes()


    def _check_attributes(self):

        list_attr = ('PARAMS', 'VARIABS', 'VARIABS_TIME')
        assert all(hasattr(self, attr) for attr in list_attr), (
            'Children of `Component` must implement all of '
            '%s'%', '.join(list_attr))


    def get_params(self):

        return [getattr(self, param_name)
                for param_name in self.PARAMS
                if hasattr(self, param_name)]


    def get_params_dict(self, attr=tuple()):

        attr = tuple(attr) if isinstance(attr, str) else attr

        param_objs = self.get_params()

        if len(attr) == 1:
            return [getattr(par, attr[0])
                    for par in param_objs]
        elif len(attr) == 2:
            return {getattr(par, attr[0]): getattr(par, attr[1])
                    for par in param_objs}
        else:
            return param_objs


    def _get_constraint_combinations(self):
        '''
        Return all relevant constraint combinations for this component.

        1. Generates full cross-product of all binding or non-binding
           inequality constraints.
        2. Filters the resulting table based on its `_MUTUALLY_EXCLUSIVE`
           dictionary.

        Returns
        -------
        pandas.DataFrame
            Relevant combinations of binding/non-binding inequality
            constraints for this component.

        '''

        constrs_cols_neq = [cstr.col for cstr in self.get_constraints()
                            if not cstr.is_equality_constraint]

        mut_excl_cols = self.get_mutually_exclusive_cstrs()

        ncombs = 2**len(constrs_cols_neq)

        logger.info('*'*30 + self.name + '*'*30)
        logger.info(('Component %s: Generating df_comb with length %d...'
                        )%(self.name, ncombs))
        bools = [[False, True] for cc in constrs_cols_neq]
        df_comb = pd.DataFrame(itertools.product(*bools),
                               columns=constrs_cols_neq, dtype=bool)
        logger.info('...done.')

        df_comb = filter_constraint_combinations(df_comb, mut_excl_cols)

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


    def get_constraints(self, by_slot=True, names=False, comp_names=False):
        '''
        Returns
        -------
        list of constraint objects

        TODO: names=True only keeps single time slot

        '''

        constrs = {} if names else []

        attrs = [(key, attr) for key, attr in vars(self).items() if
                 key.startswith('cstr_')]  # naming convention of constraint attrs;
                                           # values are dicts, so we can't check
                                           # isinstance(attr, Constraint)

        for key, attr in attrs:
            key = (self.name, key) if comp_names else key
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

        # sort by name, otherwise the order is not consistent between runs
        # TODO: implemented constraint iterator collection which is
        # instantiated by components; replaces this whole method
        if len(constrs) > 1:
            if not comp_names:
                constrs = list(zip(*sorted((cstr.base_name, cstr)
                               for cstr in constrs)))[1]

        return constrs



    def get_mutually_exclusive_cstrs(self):
        '''
        Time dependent mutually inclusive constraints.
        '''

        list_col_names = []
        for mename, me in self.MUTUALLY_EXCLUSIVE.items():

            ccb = CstrCombBase(mename, me, list(self.slots.values()),
                               self.get_constraints(by_slot=False, names=True))

            list_col_names += ccb.gen_col_combs()

        return list_col_names


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

    def get_component_hash_name(self):


        hash_input = sorted(
        [self.name] +
        list(map(lambda x: x.name, self.get_params())) +
        list(map(lambda x: x.name, self.get_variabs())) +
        list(map(lambda x: '%s_%s'%(x.expr, x.mlt), self.get_constraints()))
        )

        logger.debug('Generating component hash.')

        return md5(str(hash_input).encode('utf-8')).hexdigest()





