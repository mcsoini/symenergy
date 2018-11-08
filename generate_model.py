#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:33:36 2018

@author: user
"""

import symenergy.core.model as model
from symenergy.auxiliary.parallelization import parallelize_df
from importlib import reload

nthreads=7
reload(model)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model(solve=True, nthreads=7):

    m = model.Model(curtailment=False, nthreads=nthreads)

    self = m

    m.add_slot(name='day', load=4.5, vre=3)
    m.add_slot(name='night', load=5, vre=0.5)

    m.add_plant(name='n', vc0=1, vc1=None, slots=m.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m.add_plant(name='g', vc0=2, vc1=0, slots=m.slots)

    m.add_storage(name='phs',
                  eff=0.75,
                  slots=m.slots,
                  capacity=0.5,
                  energy_capacity=1,
                  slots_map={'day': 'chg',
                             'night': 'dch'
                             })
    if solve:
        m.generate_solve()

    return m

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model_three(solve=True, nthreads=7):

    m_three = model.Model(curtailment=True, nthreads=nthreads)

    self = m_three

    m_three.add_slot(name='day', load=4.5, vre=3)
    m_three.add_slot(name='night', load=5, vre=0.5)
    #
    m_three.add_plant(name='n', vc0=1, vc1=0, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='c', vc0=1, vc1=0, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='g', vc0=2, vc1=0, slots=m_three.slots)

    m_three.add_storage(name='phs',
                      eff=0.75,
                      slots=m_three.slots,
                      capacity=0.5,
                      energy_capacity=1,
                      slots_map={'day': 'chg',
                                 'night': 'dch'
                                 })

    if solve:
        m_three.generate_solve()

    return m_three

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model_three_curt():

    import time

    t = time.time()
    m_three = model.Model(curtailment=True, nthreads=3)

    self = m_three

    m_three.add_slot(name='day', load=4.5, vre=3)
    m_three.add_slot(name='night', load=5, vre=0.5)
    #
    m_three.add_plant(name='n', vc0=1, vc1=None, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='c', vc0=1, vc1=0, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='g', vc0=2, vc1=0, slots=m_three.slots)

    m_three.add_storage(name='phs',
                      eff=0.75,
                      slots=m_three.slots,
                      capacity=0.5,
                      energy_capacity=1,
                      slots_map={'day': 'chg',
                                 'night': 'dch'
                                 })

    m_three.generate_solve()

    print('Time: ', time.time() - t)

    return m_three

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model_simple():

    m_simple = model.Model(curtailment=True, nthreads=7)

    m_simple.add_slot(name='day', load=4.5, vre=3)

    m_simple.add_plant(name='n', vc0=1, vc1=0, slots=m_simple.slots, capacity=3)
    m_simple.add_plant(name='g', vc0=2, vc1=0, slots=m_simple.slots)

    m_simple.generate_solve()

    return m_simple

