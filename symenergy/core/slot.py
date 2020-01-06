#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Slot class. Provides a dummy slot object noneslot.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""


from symenergy.core import component
from symenergy.core.parameter import Parameter
from symenergy.auxiliary.decorators import hexdigest

from symenergy import _get_logger

logger = _get_logger(__name__)


class SlotBlock(component.Component):
    '''
    SlotBlock instances can are optional to define more complex model 
    time structures. Slot instances are assigned to SlotBlock instances.

    Parameters
    ----------
    name : str
        name of the slot block
    repetitions : int
        number of repetitions of the block. For example, a block with
        ``repetitions=3`` and two slots `s1` and `s2` corresponds to the
        pattern `s1, s2, s1, s2, s1, s2` 
    '''


    variabs = []
    variabs_time = []
    mutually_exclusive = {}

    def __init__(self, name, repetitions):

        super().__init__(name)

        self.name = name
        self.rp = self.parameters.append(Parameter('%s_rp'%self.name, noneslot,
                                                   repetitions))


class Slot(component.Component):
    '''
    Parameters
    ----------
    name : str
        name of the time slot
    load : float
        power demand (units MW, default 0)
    vre : float
        variable renewable power production (units MW, default 0)
    weight : float
        Optional time slot duration (units hours, default 1)
    block : str
        time slot block this slot is assigned to (must be one of the keys
        of the `Model.slot_blocks()` attribute)
    '''

    variabs = []
    variabs_time = []

    mutually_exclusive = {}

    def __init__(self, name, load=0, vre=None, weight=1, block=None):

        assert isinstance(name, str), 'Slot name must be string.'

        super().__init__(name)

        lst_par = [('l', load), ('vre', vre), ('w', weight)]
        for param_name, param_val in lst_par:
            self._add_parameter(param_name, param_val, self)

        self.block = block


    @hexdigest
    def _get_hash_name(self):
        '''
        Add block hash to slot hash, otherwise the time slot allocation to
        blocks would not affect the component hashes.
        '''

        hash_input_0 = super()._get_hash_name()
        hash_input = self.block._get_hash_name() if self.block else ''

        return hash_input + hash_input_0


class NoneSlot():
    '''
    Singleton class.
    '''

    def __init__(self):
        self.name = 'none'

    def __repr__(self):
        return 'NoneSlot id=%d'%id(self)

noneslot = NoneSlot()

