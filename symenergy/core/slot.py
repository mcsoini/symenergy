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

class Slot(component.Component):
    '''
    Doesn't know about plants.
    '''

    PARAMS = []
    PARAMS_TIME = ['vre', 'l']
    VARIABS = []
    VARIABS_TIME = []
    MULTIPS = []
    
    VARIABS_POSITIVE = []

    MAP_CAPACITY = {}

    MUTUALLY_EXCLUSIVE = {}

    def __init__(self, name, load, vre, weight=1):

        super().__init__(name)
#        self.name = name

        self.l = Parameter('l_%s'%self.name, self, load)
        self.vre = Parameter('vre_%s'%self.name, self, vre)

        self.weight = weight

    def __repr__(self):

        return 'Slot %s'%str(self.name)# + (', weight %s'%self.weight)

    def get_component_hash_name(self):

        hash_name_0 = super().get_component_hash_name()
        hash_input = ['{:.20f}'.format(self.weight)]

        logger.debug('Generating time slot hash.')

        return md5(str(hash_input + [hash_name_0]).encode('utf-8')).hexdigest()

class NoneSlot():
    
    def __init__(self):
        self.name = str(None)

    def __repr__(self):
        return 'Slot %s'%str(self.name)

noneslot = NoneSlot()

