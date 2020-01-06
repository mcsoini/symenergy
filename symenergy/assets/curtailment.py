"""
Contains the Curtailment class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import symenergy.core.asset as asset
from symenergy.core.slot import Slot, noneslot

class Curtailment(asset.Asset):

    variabs = []
    variabs_time = ['p']

    mutually_exclusive = {}

    def __init__(self, name, slots=None):

        super().__init__(name)

        self.slots = slots if slots else noneslot
        self._init_symbol_operation('p')
        self._init_cstr_positive('p')

