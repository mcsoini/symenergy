#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Tests for the :func:`symenergy.core.assets.storage.Storage._init_prev_slot`
method.

'''

import unittest

from symenergy.core import model
from symenergy import _get_logger

logger = _get_logger('ERROR')

#%%

class ModelMaker():
    '''

    '''

    def __init__(self, nslots):

        self.m = model.Model(curtailment=False, nthreads=2)
        self.nslots = nslots

        if isinstance(self.nslots, int):
            self.add_slots()
        else:
            self.add_slot_blocks()

        self.add_storage()


    def add_slots(self):

        for nslot in range(self.nslots):

            self.m.add_slot(name='s{:d}'.format(nslot), load=1, vre=1)

    def add_slot_blocks(self):

        for nslotblock in range(max(sl[-1] for sl in self.nslots) + 1):
            self.m.add_slot_block(name='b{:d}'.format(nslotblock),
                                  repetitions=3)

        for nslot, nslotblock in self.nslots:

            self.m.add_slot(name='s{:d}'.format(nslot),
                            block='b{:d}'.format(nslotblock),
                            load=1, vre=1)


    def add_storage(self):

        self.m.add_storage(name='storage', eff=0.9)

    def __call__(self):

        return self.m


class TestNoBlocks(unittest.TestCase):

    def test_three_two_blocks(self):

        m = ModelMaker(nslots=[(0, 0), (1, 0), (2, 1)])()
        store = m.storages['storage']

        s0 = m.slots['s0']
        s1 = m.slots['s1']
        s2 = m.slots['s2']

        _dict_prev_slot_expect = {s0: s1, s1: s0, s2: s2}

        assert store._dict_prev_slot == _dict_prev_slot_expect


    def test_three_no_blocks(self):

        m = ModelMaker(nslots=3)()
        store = m.storages['storage']

        s0 = m.slots['s0']
        s1 = m.slots['s1']
        s2 = m.slots['s2']

        _dict_prev_slot_expect = {s0: s2, s1: s0, s2: s1}

        assert store._dict_prev_slot == _dict_prev_slot_expect

    def test_one_no_blocks(self):

        m = ModelMaker(nslots=1)()
        store = m.storages['storage']

        s0 = m.slots['s0']

        _dict_prev_slot_expect = {s0: s0}

        assert store._dict_prev_slot == _dict_prev_slot_expect


    def test_four_two_blocks(self):

        m = ModelMaker(nslots=[(0, 0), (1, 0), (2, 1), (3, 1)])()
        store = m.storages['storage']

        s0 = m.slots['s0']
        s1 = m.slots['s1']
        s2 = m.slots['s2']
        s3 = m.slots['s3']

        _dict_prev_slot_expect = {s0: s1, s1: s0, s2: s3, s3: s2}

        assert store._dict_prev_slot == _dict_prev_slot_expect


    def test_three_three_blocks(self):

        m = ModelMaker(nslots=[(0, 0), (1, 1), (2, 2)])()

        store = m.storages['storage']

        s0 = m.slots['s0']
        s1 = m.slots['s1']
        s2 = m.slots['s2']

        _dict_prev_slot_expect = {s0: s0, s1: s1, s2: s2}

        assert store._dict_prev_slot == _dict_prev_slot_expect


if __name__ == '__main__':


    unittest.main()


