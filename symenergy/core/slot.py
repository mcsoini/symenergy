#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Slot class. Provides a dummy slot object noneslot.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

from hashlib import md5

from collections import namedtuple

from symenergy.core import component
from symenergy.core.parameter import Parameter

from symenergy import _get_logger

logger = _get_logger(__name__)

SlotBlock = namedtuple('SlotBlock', ['name', 'repetitions'])

class Slot(component.Component):
    '''
    Doesn't know about plants.
    '''

    PARAMS = ['vre', 'l', 'w']
    VARIABS = []
    VARIABS_TIME = []

    VARIABS_POSITIVE = []

    MAP_CAPACITY = {}

    MUTUALLY_EXCLUSIVE = {}

    def __init__(self, name, load, vre, weight=1, block=None, repetitions=1):

        assert isinstance(name, str), 'Slot name must be string.'

        super().__init__(name)
#        self.name = name

        self.l = Parameter('l_%s'%self.name, self, load)
        self.vre = Parameter('vre_%s'%self.name, self, vre)
        self.w = Parameter('w_%s'%self.name, self, weight)

        self.block = block
        self.repetitions = repetitions


    def __repr__(self):

        return 'Slot %s'%str(self.name)# + (', weight %s'%self.weight)


    def get_component_hash_name(self):

        hash_name_0 = super().get_component_hash_name()
        hash_input = ['{}-{:d}'.format(self.block, self.repetitions)]

        logger.debug('Generating time slot hash.')

        return md5(str(hash_input + [hash_name_0]).encode('utf-8')).hexdigest()


class NoneSlot():
    
    def __init__(self):
        self.name = str(None)

    def __repr__(self):
        return 'Slot %s'%str(self.name)

noneslot = NoneSlot()

