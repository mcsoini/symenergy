#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the symenergy Component class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import itertools
import pandas as pd
from symenergy.auxiliary.constrcomb import CstrCombBase
from symenergy.auxiliary.constrcomb import make_constraint_combination_table

from symenergy.core.parameter import Parameter
from symenergy.core.collections import VariableCollection
from symenergy.core.collections import ConstraintCollection
from symenergy.core.collections import ParameterCollection
from symenergy.auxiliary.decorators import hexdigest

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

            if isinstance(val, Parameter):  # -> global slot weight parameters
                newpar = val
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

        logger.info('Generating constraint combinations for "%s"' % self.name)

        filt = dict(is_equality_constraint=False)
        constrs_cols_neq = self.constraints.tolist('col', **filt)
        constrs_cols_neq = {col: i for i, col in enumerate(constrs_cols_neq)}

        if constrs_cols_neq:

            mut_excl_cols = self._get_mutually_exclusive_cstrs()
            # translate to column id
            mut_excl_cols = [[(constrs_cols_neq[col[0]], col[1])
                             for col in comb] for comb in mut_excl_cols]


            df_comb = make_constraint_combination_table(len(constrs_cols_neq),
                                                mutually_excl=mut_excl_cols)
            df_comb = df_comb.rename(columns=dict(enumerate(constrs_cols_neq)))

            return df_comb

        else:
            return pd.DataFrame()


    def _get_mutually_exclusive_cstrs(self):
        '''
        Time dependent mutually inclusive constraints.
        '''

        list_col_names = []
        if __name__ == '__main__':
            mename, me = list(self.mutually_exclusive.items())[1]
        for mename, me in self.mutually_exclusive.items():

            ccb = CstrCombBase(mename, me, list(self.slots.values()),
                       self.constraints.to_dict(dict_struct={'name_no_comp': {'slot': ''}})
                       )

            list_col_names += ccb.gen_col_combs()

        return list_col_names

    @hexdigest
    def _get_hash_name(self):

        return (str(self.name)
                  + str(self.constraints('expr'))
                  + str(self.parameters._get_hash_name())
                  + str(self.variables('name')))


    def __repr__(self):

        return '{} `{}`'.format(self.__class__.__name__, self.name)



