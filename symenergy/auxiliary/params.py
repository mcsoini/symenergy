#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains a base class for parameter management.
"""

from collections import UserDict

class RcParams(UserDict):
    '''
    Base class to define dictionaries with restricted keys and values.

    Children must implement "items_opts" and "items_default"
    '''

    items_opts = None
    items_default = None

    def __init__(self):

        for attr in ['items_opts', 'items_default']:
            assert isinstance(getattr(self, attr, None), dict), (
                    'Children of RcParams must implement dictionary class '
                    f'attribute {attr}')

        super().__init__(self.items_default.copy())


    def __setitem__(self, key, item):

        assert key in self.items_opts, 'RcParams: Unknown key %s ' % key
        value = [val for val in self.items_opts[key]['values'] if item is val]
        if value:
            self.data[key] = item
            return

        type_ = [tp for tp in self.items_opts[key]['types']
                 if isinstance(item, tp)]
        if type_:
            type_ = type_[0]
            cond = self.items_opts[key]['types'][type_]
            mssg = (f'RcParams: Value for {key} of type {type_} must '
                    f'be %s; received {item}')
            if  cond == '>=0':
                assert item >= 0, mssg % cond


            self.data[key] = item
            return

        raise ValueError(f'RcParams: Invalid value "{item}" for parameter '
                         f'{key}; must be ' + self.items_opts[key]['cond'])
