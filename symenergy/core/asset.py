#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Asset class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""


import sympy as sp
import itertools
from orderedset import OrderedSet

import symenergy.core.component as component
from symenergy.core.constraint import Constraint
from symenergy.core.parameter import Parameter
from symenergy.core.variable import Variable
from symenergy.core.slot import noneslot
from symenergy.auxiliary.decorators import hexdigest


from symenergy import _get_logger

logger = _get_logger(__name__)

class UnexpectedSymbolError(Exception):

    def __init__(self, res):

        message = ('Method called with unexpected '
                   'variable/parameter %s'%(res))

        super().__init__(message)

def _expand_class_attrs(cls):
    cls._add_default_cap_constr_sgn()
    return cls


@_expand_class_attrs
class Asset(component.Component):
    '''
    Mixin class containing shared methods of plants and storage.

    Not instantiated directly.
    '''

    # capacity class map
    # C -> all power and retired capacity
    # E -> storage energy capacity and block transfer
    map_capacity = {'C': ['p', 'pchg', 'pdch', 'C_ret'],
                    'E': ['e', 'et']}

    # positive and or variable are capacity constraint; only for `et`, since it
    # can be both pos and neg
    capacity_constraint_sign = {'et': [+1, -1]}  # default is [+1]


    def __init__(self, name):

        super().__init__(name)

        self.params = []


    @property
    def cc(self):

        return self._cc


    @cc.setter
    def cc(self, cc):

        self._cc = sp.simplify(cc)


    @classmethod
    def _add_default_cap_constr_sgn(cls):

        list_vars = itertools.chain.from_iterable(cls.map_capacity.values())
        dict_cstr_sgn = dict(itertools.product(list_vars, [[+1]]))
        dict_cstr_sgn.update(cls.capacity_constraint_sign)
        cls.capacity_constraint_sign = dict_cstr_sgn


    def get_constrained_variabs(self):
        '''
        Keyword argument:
            * cap -- string, capacity name as specified in self.map_capacity

        Returns:
            * list of tuples [(parameter object capacity,
                               list relevant var strings)]
        '''

        cap_var = []

        c_name, variabs = list(self.map_capacity.items())[0]
        for c_name, variabs in self.map_capacity.items():
            if hasattr(self, c_name):
                cap = getattr(self, c_name)

                constr_var = []

                for var in variabs:
                    if hasattr(self, var):
                        variabs = getattr(self, var, None)

                        constr_var += list(variabs.values())

                cap_var.append((cap, constr_var))

        return cap_var


    def expr_func_capacity(self, slot, var_name, capacity_name, sgn):
        # define expression
        var = getattr(self, var_name)[slot]
        cap = getattr(self, capacity_name).symb

        # subtract retired capacity if applicable
        if (hasattr(self, capacity_name + '_ret')
            # ... not for retired capacity constraint
            and not capacity_name + '_ret' == var_name):
            cap -= getattr(self, capacity_name + '_ret')[noneslot]

        return sgn * var - cap


    def _init_single_cstr_capacity(self, var_name, capacity_name, sgn):

        slot_objs = (self.slots.values() if self.get_flag_timedep(var_name)
             else [noneslot])

        var_attr = getattr(self, var_name)



        for slot in OrderedSet(slot_objs) & OrderedSet(var_attr):
            base_name = '%s_%s_cap%s_%s'%(self.name, var_name,
                                           {-1: 'neg', +1: ''}[sgn],
                                           capacity_name)

            cstr = Constraint(base_name=base_name, slot=slot,
                              expr_func=self.expr_func_capacity,
                              expr_args=(slot, var_name, capacity_name, sgn),
                              comp_name=self.name,
                              is_capacity_constraint=True,
                              var_name=str(var_attr[slot]))

            self.constraints.append(cstr)


    def _init_cstr_capacity(self, capacity_name):
        '''
        Instantiates a dictionary {slot symbol: Constraint}.

        Applies to power and capacity retirement, both of which are smaller
        than the initially installed capacity.
        '''

        if not capacity_name in self.map_capacity:
            raise UnexpectedSymbolError(capacity_name)

        list_var_names = self.map_capacity[capacity_name]
        list_var_names = [var for var in list_var_names if hasattr(self, var)]

        for var_name in list_var_names:  # loop over constrained variables

            for sgn in self.capacity_constraint_sign[var_name]:

                self._init_single_cstr_capacity(var_name, capacity_name, sgn)




    def expr_func_positive(self, slot, var_attr):
        return var_attr[slot]


    def _init_cstr_positive(self, variable):
        '''
        Instantiates a dictionary {slot symbol: Constraint}.
        '''

        slot_objs = (self.slots.values()
                     if self.get_flag_timedep(variable)
                     else [noneslot])

        var_attr = getattr(self, variable)

        self.variables.to_dict({'slot': ''})

        for slot in OrderedSet(slot_objs) & OrderedSet(var_attr):

            base_name = '%s_pos_%s'%(self.name, variable)

            cstr = Constraint(base_name=base_name, slot=slot,
                              expr_func=self.expr_func_positive,
                              var_name=str(var_attr[slot]),
                              is_positivity_constraint=True,
                              comp_name=self.name, expr_args=(slot, var_attr))

            self.constraints.append(cstr)


    def get_flag_timedep(self, variable):
        '''
        TODO: The first case should depend on the chg/dch slots.
        '''


        if variable in (OrderedSet(self.variabs)
                        & OrderedSet(self.variabs_time)):
            # the variable is defined for all time slots only if there are
            # two or more time slots (used for stored energy)

            flag_timedep = len(self.slots) >= 2

        elif variable in OrderedSet(self.variabs) | OrderedSet(self.variabs_time):
            flag_timedep = variable in self.variabs_time

        else:
            raise UnexpectedSymbolError(variable)

        logger.info('Variable %s has time dependence %s'%(variable,
                                                          flag_timedep))

        return flag_timedep


    def _init_symbol_operation(self, variable, slotsslct=None):
        '''
        Sets operational variables, i.e. power (generation,
        charging, discharging) and stored energy.

        Parameters
        ----------
        variable: str
            collective variable name to be added
        slotsslct: list of strings
            list of time slot names for which the variable is to be defined;
            defaults to all time slots

        '''

        flag_timedep = self.get_flag_timedep(variable)

        if not flag_timedep:
            self.variables.append(Variable(variable, noneslot, self.name))

        else:
            if slotsslct:
                slots = [slot for slot_name, slot in self.slots.items()
                          if slot_name in slotsslct]
            else:
                slots = list(self.slots.values())
            for slot in slots:
                self.variables.append(Variable(variable, slot, self.name))

        setattr(self, variable, self.variables.to_dict({'slot': 'symb'},
                                                       name_no_comp=variable))


    def init_symbols_costs(self):
        ''' Overridden by children, if applicable. '''


    def _subs_cost(self, symb, *args, **kwargs):
        ''' Overridden by children, if applicable. '''

        return symb


    @hexdigest
    def _get_hash_name(self):

        hash_name_0 = super()._get_hash_name()
        # adding slots
        hash_input = str(tuple(map(lambda x: '%s'%(x.name),
                                   self.slots.values())))

        logger.debug('Generating asset hash.')

        return hash_input + hash_name_0

