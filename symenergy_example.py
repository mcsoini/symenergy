#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:38:17 2018

@author: user
"""

%reset -f

from importlib import reload
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import symenergy.core.model as model
import symenergy.evaluator.evaluator as evaluator
import pyAndy.core.plotpage as pltpg

reload(model)

m = model.Model(curtailment=True, nthreads=7)

self = m

m.add_slot(name='day', load=4.5, vre=3)
m.add_slot(name='night', load=5, vre=0.5)

m.add_plant(name='n', vc0=1, vc1=0, slots=m.slots, capacity=3,
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

m.generate_solve()


# %%
reload(evaluator)

m.comps['n'].vc1.value = 0
m.comps['n'].vc0.value = 10
m.comps['n'].fcom.value = 9
m.comps['n'].C.value = 5000

m.comps['g'].vc1.value = 0
m.comps['g'].vc0.value = 160

m.slots['day'].l.value = 6500
m.slots['night'].l.value = 5200
m.slots['day'].vre.value = m.slots['day'].l.value + m.slots['night'].l.value
m.slots['night'].vre.value = 0

m.storages['phs'].eff.value = 0.75
m.storages['phs'].C.value = 3250
m.storages['phs'].E.value = 10000

x_vals = {
         m.vre_scale: np.linspace(0, 1, 101),
         m.comps['phs'].C: np.linspace(0, 3250, 2),
        }

ev = evaluator.Evaluator(m, x_vals)

ev.get_evaluated_lambdas(skip_multipliers=True)

ev.expand_to_x_vals()

ev.enforce_constraints()

ev.init_cost_optimum()

ev.map_func_to_slot()

ev.drop_non_optimal_combinations()

ev.build_supply_table()

# %%

ev.line_plot(all_labels=False)

ev.supply_plot(ind_axx=['vre_scale'],
               ind_plty=['C_phs'])


