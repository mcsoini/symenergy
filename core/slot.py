#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Slot class. Provides a dummy slot object noneslot.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""


import symenergy.core.component as component
from symenergy.core.parameter import Parameter

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

        self.name = name

        self.l = Parameter('l_%s'%self.name, self, load)
        self.vre = Parameter('vre_%s'%self.name, self, vre)

        self.weight = weight

    def __repr__(self):

        return 'Slot %s'%str(self.name) + (', weight %s'%self.weight)

class NoneSlot():
    
    def __init__(self):
        self.name = str(None)

    def __repr__(self):
        return 'Slot %s'%str(self.name)

noneslot = NoneSlot()

