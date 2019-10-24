#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Slot class. Provides a dummy slot object noneslot.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

from hashlib import md5

from symenergy.core import component
from symenergy.core.parameter import Parameter

from symenergy import _get_logger

logger = _get_logger(__name__)


class SlotBlock(component.Component):

    VARIABS = []
    VARIABS_TIME = []
    MAP_CAPACITY = {}
    MUTUALLY_EXCLUSIVE = {}

    def __init__(self, name, repetitions):

        super().__init__(name)

        self.name = name
        self.rp = self.parameters.append(Parameter('rp', noneslot, repetitions))

    def _get_hash_name(self):

        hash_input = [self.name]
        logger.debug('Generating time slot block hash.')

        return md5(str(hash_input).encode('utf-8')).hexdigest()


class Slot(component.Component):
    '''
    '''

    VARIABS = []
    VARIABS_TIME = []

    MAP_CAPACITY = {}

    MUTUALLY_EXCLUSIVE = {}

    def __init__(self, name, load, vre, weight=1, block=None, repetitions=1):

        assert isinstance(name, str), 'Slot name must be string.'

        super().__init__(name)

        self.l = self.parameters.append(Parameter('l', self, load))
        self.vre = self.parameters.append(Parameter('vre', self, vre))

        if isinstance(weight, int):
            self.w = self.parameters.append(Parameter('w', self, weight))
        elif isinstance(weight, Parameter):
            self.w = self.parameters.append(weight)

        self.block = block
        self.repetitions = repetitions


    def _get_component_hash_name(self):

        hash_name_0 = super()._get_component_hash_name()
        hash_input = self.block._get_hash_name() if self.block else ''

        logger.debug('Generating time slot hash.')

        return md5(str(hash_input + hash_name_0).encode('utf-8')).hexdigest()


class NoneSlot():
    '''
    Singleton class.
    '''

    def __init__(self):
        self.name = 'none'

    def __repr__(self):
        return 'NoneSlot id=%d'%id(self)

noneslot = NoneSlot()

