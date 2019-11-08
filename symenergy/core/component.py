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

from symenergy.core.parameter import Parameter
from symenergy.core.collections import VariableCollection
from symenergy.core.collections import ConstraintCollection
from symenergy.core.collections import ParameterCollection

from symenergy import _get_logger

logger = _get_logger(__name__)


class Component():
    '''
    Base class for components.

    This serves as parent class for:

    - time slots (:class:`symenergy.core.slot.Slot`)
    - time slot_blocks (:class:`symenergy.core.slot.SlotBlock`)
    - assets (:class:`symenergy.core.asset.Asset`)

    It is not instantiated directly.

    '''

    map_capacity = {}
    variabs = []
    variabs_time = []

    def __init__(self, name):

        self.name = name

        self.constraints = ConstraintCollection('%s-constraints'%(self.name))
        self.parameters = ParameterCollection('%s-parameters'%(self.name))
        self.variables = VariableCollection('%s-variables'%(self.name))


    def _add_parameter(self, name, val, slot):
        ''''Combines the definition of various parameters.'''

        if val:
            if self.name != slot.name:
                parname = '%s_%s'%(name, self.name)
            else:  # self is slot
                parname = name

            if isinstance(val, Parameter):
                newpar = val  # -> for common weight parameters of slots
            elif isinstance(val, (float, int)):
                newpar = Parameter(parname, slot, val)

            setattr(self, name, self.parameters.append(newpar))

            if name in self.map_capacity:
                self._init_cstr_capacity(name)


    def _reinit_all_constraints(self):
        '''
        Re-initialize all constraint expressions.

        This is necessary if parameter values are frozen or the values of
        frozen parameters are changed.
        '''

        for cstr in self.constraints():
            cstr.make_expr()


    def freeze_all_parameters(self):

        for param in self.get_params():
            param._freeze_value()




    def _get_constraint_combinations(self):
        '''
        Return all relevant constraint combinations for this component.

        1. Generates full cross-product of all binding or non-binding
           inequality constraints.
        2. Filters the resulting table based on its `mutually_exclusive`
           dictionary.

        Returns
        -------
        pandas.DataFrame
            Relevant combinations of binding/non-binding inequality
            constraints for this component.

        '''

        filt = dict(is_equality_constraint=False)
        constrs_cols_neq = self.constraints.tolist('col', **filt)

        mut_excl_cols = self._get_mutually_exclusive_cstrs()

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


    def _get_mutually_exclusive_cstrs(self):
        '''
        Time dependent mutually inclusive constraints.
        '''

        list_col_names = []
        for mename, me in self.mutually_exclusive.items():

            ccb = CstrCombBase(mename, me, list(self.slots.values()),
                       self.constraints.to_dict(dict_struct={'name_no_comp': {'slot': ''}})
                       )

            list_col_names += ccb.gen_col_combs()

        return list_col_names


    def _get_component_hash_name(self):


        hash_input = sorted(
        [self.name]
#        list(map(lambda x: x.name, self.get_params())) +
#        list(map(lambda x: x.name, self.get_variabs())) +
#        list(map(lambda x: '%s_%s'%(x.expr, x.mlt), self.get_constraints()))
        )

        logger.debug('Generating component hash.')

        return md5(str(hash_input).encode('utf-8')).hexdigest()


    def __repr__(self):

        return '{} `{}`'.format(self.__class__.__name__, self.name)



