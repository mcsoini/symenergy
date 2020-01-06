#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Plant class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import sympy as sp

import symenergy.core.asset as asset
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter

class Plant(asset.Asset):
    '''
    Parameters
    ----------

    name : str
        name of the power plant
    vc0 : float
        Optional constant coefficient of the linear variable cost supply curve
    vc1 : float
        Optional slope of variable cost supply curve
    fcom : float
        Optional O&M fixed cost; only relevant if ``cap_ret`` is ``True``
    capacity : float
        Optional available installed capacity; this limits the power
        output. Power production is unconstrained if this argument is not
        set.
    cap_ret : bool
        Optional: power capacity can be retired if True


    **Example:**

    .. code-block:: python

        >>> from symenery.core import model
        >>> m = model.Model()
        >>> m.add_slot('t0', load=10)
        >>> m.add_slot('t1', load=10)
        >>> m.add_plant('n', vc0=1, vc1=2)

    The cost supply curve is determined by a linear function of the power production
    with the cost coefficients ``vc0` and `vc1``. For each time slot, 
    it's expression is given by

    .. code-block:: python

        >>> m.plants['n'].vc
        {Slot `t0`: w_none*(n_p_t0*vc1_n_none + vc0_n_none),
         Slot `t1`: w_none*(n_p_t1*vc1_n_none + vc0_n_none)}

    Consequently, the power plant cost component is quadratic:

    .. code-block:: python

        >>> m.plants['n'].cc
        w_none*(n_p_t0**2*vc1_n_none + 2*n_p_t0*vc0_n_none + n_p_t1**2*vc1_n_none + 2*n_p_t1*vc0_n_none)/2

    '''
    variabs = ['C_ret']
    variabs_time = ['p']

    # mutually exclusive constraint combinations
    mutually_exclusive = {
# =============================================================================
# TODO: This needs to be fixed: C_ret defined for Noneslot
#         'Power plant retirement not simult. max end zero':
#             (('pos_C_ret', 'this', True), ('C_ret_cap_C', 'this', True)),
# =============================================================================
        'Power plant output not simult. max end zero':
            (('pos_p', 'this', True), ('p_cap_C', 'this', True))

        # C_ret max --> no output
        }

    def __init__(self, name, vc0=None, vc1=None,
                 fcom=None, slots=None, capacity=False, cap_ret=False):

        super().__init__(name)

        self.slots = slots if slots else noneslot

        self._init_symbol_operation('p')
        self._init_cstr_positive('p')

        if cap_ret:
            # needs to be initialized before _init_cstr_capacity('C')!
            self._init_symbol_operation('C_ret')
            self._init_cstr_positive('C_ret')

        lst_par = [('vc0', vc0), ('vc1', vc1), ('fcom', fcom), ('C', capacity)]
        for param_name, param_val in lst_par:
            self._add_parameter(param_name, param_val, noneslot)

        self._init_cost_component()


    def _init_cost_component(self):
        '''
        Set constant and linear components of variable costs.

        If a fixed O&M cost parameter is defined, adds the corresponding
        fixed cost.
        '''

        def get_vc(slot):
            cost = 0
            if hasattr(self, 'vc0'):
                cost += self.vc0.symb
            if hasattr(self, 'vc1'):
                cost += self.vc1.symb * self.p[slot]

            return cost

        self.vc = {slot: get_vc(slot) * slot.w.symb
                   * (slot.block.rp.symb if slot.block else 1)
                   for slot in self.slots.values()}

        self.cc = sum(sp.integrate(vc, self.p[slot])
                      for slot, vc in self.vc.items())

        if hasattr(self, 'fcom'):

            cc_fcom = self.C.symb * self.fcom.symb

            if hasattr(self, 'C_ret'):
                cc_fcom -= self.C_ret[noneslot] * self.fcom.symb

            self.cc += cc_fcom

