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
solve=True
reload(model)

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model_weights(solve=True, nthreads=7):

    m = model.Model(curtailment=True, nthreads=nthreads)

    self = m

    m.add_slot(name='day', weight=18, load=4.5, vre=3)
    m.add_slot(name='night', weight=6, load=5, vre=0.5)

    m.add_plant(name='n', vc0=1, vc1=None, slots=m.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m.add_plant(name='g', vc0=2, vc1=None, slots=m.slots)

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

def get_model(solve=True, nthreads=7):

    m = model.Model(curtailment=True, nthreads=nthreads)

    self = m

    m.add_slot(name='day', load=4.5, vre=3)
    m.add_slot(name='night', load=5, vre=0.5)

    m.add_plant(name='n', vc0=1, vc1=None, slots=m.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m.add_plant(name='g', vc0=2, vc1=None, slots=m.slots)

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

def get_model_illustration(solve=True, nthreads=7):

    m = model.Model(curtailment=False)

    m.add_slot(name='day', load=6500, vre=6500)

    m.add_plant(name='n', vc0=10, vc1=0.0, slots=m.slots, capacity=3500, cap_ret=False)
    m.add_plant(name='g', vc0=90, vc1=0.0, slots=m.slots)

    if solve:
        m.generate_solve()

    return m


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model_lin(solve=True, nthreads=7):

    m = model.Model(curtailment=True, nthreads=nthreads)

    self = m

    m.add_slot(name='day', load=4.5, vre=3)
    m.add_slot(name='night', load=5, vre=0.5)

    m.add_plant(name='n', vc0=1, vc1=None, slots=m.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m.add_plant(name='g', vc0=2, vc1=1, slots=m.slots)

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

    m_three = model.Model(curtailment=False, nthreads=nthreads)

    self = m_three

    m_three.add_slot(name='day', load=4.5, vre=3)
    m_three.add_slot(name='night', load=5, vre=0.5)
    #
    m_three.add_plant(name='n', vc0=1, vc1=None, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='c', vc0=1, vc1=1, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='g', vc0=2, vc1=1, slots=m_three.slots)

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

def get_model_three_curt(solve=True, nthreads=7):

    import time

    t = time.time()
    m_three = model.Model(curtailment=True, nthreads=nthreads)

    self = m_three

    m_three.add_slot(name='day', load=4.5, vre=3)
    m_three.add_slot(name='night', load=5, vre=0.5)
    #
    m_three.add_plant(name='n', vc0=1, vc1=None, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='c', vc0=1, vc1=1, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='g', vc0=2, vc1=1, slots=m_three.slots)

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

    print('Time: ', time.time() - t)

    return m_three


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model_three_curt_flat(solve=True, nthreads=7):

    import time

    t = time.time()
    m_three = model.Model(curtailment=True, nthreads=nthreads)

    self = m_three

    m_three.add_slot(name='day', load=4.5, vre=3)
    m_three.add_slot(name='night', load=5, vre=0.5)
    #
    m_three.add_plant(name='n', vc0=1, vc1=None, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='c', vc0=1, vc1=None, slots=m_three.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m_three.add_plant(name='g', vc0=2, vc1=None, slots=m_three.slots)

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

    print('Time: ', time.time() - t)

    return m_three

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model_simple(solve=True, nthreads=7):

    m_simple = model.Model(curtailment=True, nthreads=nthreads)

    self = m_simple

    m_simple.add_slot(name='day', load=4.5, vre=3)

    m_simple.add_plant(name='n', vc0=1, vc1=None, slots=m_simple.slots, capacity=3)
    m_simple.add_plant(name='g', vc0=2, vc1=None, slots=m_simple.slots)

    if solve:
        m_simple.generate_solve()

    return m_simple

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_model_multi_ret(solve=True, nthreads=7):

    m = model.Model(curtailment=False, nthreads=nthreads)

    self = m

    m.add_slot(name='day', load=4.5, vre=3)

    m.add_plant(name='n', vc0=1, vc1=None, slots=m.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m.add_plant(name='c', vc0=1, vc1=None, slots=m.slots, capacity=3,
                fcom=10,
                cap_ret=True
                )
    m.add_plant(name='g', vc0=2, vc1=None, slots=m.slots)

    if solve:
        m.generate_solve()

    return m

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def generate_all(nthreads):

    get_model_multi_ret(True, nthreads)
    get_model_simple(True, nthreads)
    get_model_three_curt(True, nthreads)
    get_model_three(True, nthreads)
    get_model_lin(True, nthreads)
    get_model(True, nthreads)
    get_model_weights(True, nthreads)


def generate_all_small(nthreads):

    get_model_weights(True, nthreads)
    get_model_lin(True, nthreads)
    get_model(True, nthreads)
    get_model_multi_ret(True, nthreads)
    get_model_simple(True, nthreads)
