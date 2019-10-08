#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:32:48 2019

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for constraint combination expansion.

"""

import unittest

import wrapt
import os
import shutil

import numpy as np
import pandas as pd

from symenergy.core import model
from symenergy.auxiliary.constrcomb import CstrCombBase
from symenergy import _get_logger

logger = _get_logger('ERROR')

#%%

class ModelMaker():
    '''

    '''

    def __init__(self, nslots):

        self.m = model.Model(curtailment=False, nthreads=2)
        self.nslots = nslots

        self.add_slots()
        self.add_plants()

    def add_slots(self):

        for nslot in range(self.nslots):
            self.m.add_slot(name='s%d'%nslot, load=4.5, vre=3)

    def add_plants(self):
        ''' Implemented in children. '''

    def get_model(self):

        return self.m


class ModelMakerSingleAsset(ModelMaker):

    def __init__(self, assettype, nslots):

        self.assettype = assettype
        super().__init__(nslots)


    def add_plants(self):

        if self.assettype == 'plant_no_cap':
            self.m.add_plant(name='n', vc0=1, vc1=None, slots=self.m.slots)
        elif self.assettype == 'plant':
            self.m.add_plant(name='n', vc0=1, vc1=None, slots=self.m.slots,
                             capacity=2)
        elif self.assettype == 'storage_no_cap_all_slots':
            self.m.add_storage(name='phs', eff=0.75, slots=self.m.slots)
        else:
            raise ValueError('Unknown assettype %s'%self.assettype)


class UpDown():

    def setUp_0(self):

        pass

    def tearDown_0(self):

        pass

class TestSingleAssets(unittest.TestCase, UpDown):

    def tearDown(self):

        super().tearDown_0()

    def setUp(self):

        super().setUp_0()

    def test_plant_no_cap_only(self):

        m = ModelMakerSingleAsset('plant_no_cap', 2).get_model()
        plant = m.comps['n']

        dict_me = plant.MUTUALLY_EXCLUSIVE

        mename = 'Power plant output not simult. max end zero'
        list_cstrs = dict_me[mename]
        ccb = CstrCombBase(mename, list_cstrs, list(m.slots.values()),
                     plant.get_constraints(by_slot=False, names=True))
        assert not ccb.gen_col_combs()

    def test_plant(self):

        m = ModelMakerSingleAsset('plant', 2).get_model()
        plant = m.comps['n']

        dict_me = plant.MUTUALLY_EXCLUSIVE

        mename = 'Power plant output not simult. max end zero'
        list_cstrs = dict_me[mename]
        ccb = CstrCombBase(mename, list_cstrs, list(m.slots.values()),
                     plant.get_constraints(by_slot=False, names=True))

        expect = [(('act_lb_n_pos_p_s0', True), ('act_lb_n_p_cap_C_s0', True)),
                  (('act_lb_n_pos_p_s1', True), ('act_lb_n_p_cap_C_s1', True))]

        assert expect == ccb.gen_col_combs()



    def test_storage_no_cap_all_slots(self):

        m = ModelMakerSingleAsset('storage_no_cap_all_slots', 2).get_model()
        plant = m.comps['phs']

        dict_me = plant.MUTUALLY_EXCLUSIVE

        mename = 'Empty storage stays empty w/o charging_0'
        list_cstrs = dict_me[mename]
        ccb = CstrCombBase(mename, list_cstrs, list(m.slots.values()),
                     plant.get_constraints(by_slot=False, names=True))

        expect = [(('act_lb_phs_pos_e_s0', True),
                   ('act_lb_phs_pos_pchg_s1', True),
                   ('act_lb_phs_pos_e_s1', False)),
                  (('act_lb_phs_pos_e_s1', True),
                   ('act_lb_phs_pos_pchg_s0', True),
                   ('act_lb_phs_pos_e_s0', False))]

        assert expect == ccb.gen_col_combs(), mename

        mename = 'Empty storage stays empty w/o charging_1'
        list_cstrs = dict_me[mename]
        ccb = CstrCombBase(mename, list_cstrs, list(m.slots.values()),
                     plant.get_constraints(by_slot=False, names=True))

        expect = [(('act_lb_phs_pos_e_s0', True),
                   ('act_lb_phs_pos_pchg_s1', False),
                   ('act_lb_phs_pos_e_s1', True)),
                  (('act_lb_phs_pos_e_s1', True),
                   ('act_lb_phs_pos_pchg_s0', False),
                   ('act_lb_phs_pos_e_s0', True))]

        assert expect == ccb.gen_col_combs(), mename


if __name__ == '__main__':

    unittest.main()



