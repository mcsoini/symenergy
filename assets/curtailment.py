#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Curtailment class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import symenergy.core.asset as asset
from symenergy.core.slot import Slot

class Curtailment(asset.Asset):
    '''    '''

    PARAMS = []
    PARAMS_TIME = []
    VARIABS = []
    VARIABS_TIME = ['p']

    VARIABS_POSITIVE = ['p']

    def __init__(self, name, slots=None):

        '''
        Params:
            * name --
            * vc0 --
            * vc1 --
            * fcom -- float, O&M fixed cost
            * slots -- iterable of time slot names
            * capacity --
            * cap_ret -- boolean, capacity can be retired True/False
        '''
        super().__init__()

        self.slots = slots if slots else {'0': Slot('0', 0, 0)}

        self.name = name

        self.init_symbol_operation('p')

        self.init_cstr_positive('p')

        self.init_is_positive()


if __name__ == '__main__':



    x = Curtailment('x', m.slots)

    x.p[x.slots['day']]
