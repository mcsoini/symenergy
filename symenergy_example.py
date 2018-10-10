#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:38:17 2018

@author: user
"""

%reset -f

#from sympy import Matrix, S, linsolve, symbols, lambdify, in
from importlib import reload
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from grimsel.auxiliary.aux_general import print_full

import symenergy.core.model as model
import symenergy.evaluator.evaluator as evaluator
import pyAndy.core.plotpage as pltpg

reload(model)

m = model.Model(curtailment=False)

self = m

m.add_slot(name='day', load=4.5, vre=3)
# m.add_slot(name='night', load=3, vre=1)
m.add_slot(name='evening', load=5, vre=0.5)
>>>>>>> multiple_select_x

#m.add_plant(name='b', vc0=1, vc1=0, slots=m.slots, capacity=1.5)
m.add_plant(name='n', vc0=1, vc1=0, slots=m.slots, capacity=3,
#            fcom=10,
#            cap_ret=True
            )
m.add_plant(name='g', vc0=2, vc1=0, slots=m.slots)

#m.add_storage(name='phs',
#              eff=0.75,
#              slots=m.slots,
#              capacity=0.5,
#              energy_capacity=1,
#              slots_map={'day': 'chg',
#                         'evening': 'dch'
#                         })

comp = m.comps['n']

m.init_supply_constraints()
m.init_constraint_combinations()

m.remove_mutually_exclusive_combinations()

m.delete_cached()

m.solve_all(nthreads=7)

m.filter_invalid_solutions()


# %%
reload(evaluator)

m.comps['n'].vc1.value = 0
m.comps['n'].vc0.value = 1.9 / 0.33 + 3.9
#m.comps['n'].fcom.value = 20000 / 8760 * 2
m.comps['n'].C.value = 3500

#m.comps['g'].vc1.value = 0
#m.comps['g'].vc0.value = 50 + 3

m.slots['day'].vre.value = 1
m.slots['day'].l.value = 4500
#m.slots['evening'].vre.value = 0.1
#m.slots['evening'].l.value = 3700

#m.storages['phs'].eff.value = 0.75
#m.storages['phs'].C.value = 1000
#m.storages['phs'].E.value = 20000

m.vre_scale.value = 0.5

select_x = m.vre_scale
x_vals = {m.vre_scale: np.linspace(0, 9000, 41),
          m.plants['n'].C: np.linspace(1000, 5000, 3)
          }

m.init_total_param_values()

model = m
ev = evaluator.Evaluator(model, x_vals)

self = ev

list_dep_var = (list(map(str, list(self.model.variabs.keys()) + list(self.model.multips.keys()))))
list_dep_var = ['tc'] + [v for v in list_dep_var if not 'lb_' in v]
ev.get_evaluated_lambdas(list_dep_var)

ev.expand_to_x_vals()

ev.enforce_constraints()

ev.init_cost_optimum()

ev.map_func_to_slot()

ev.drop_non_optimal_combinations()

ev.line_plot(all_labels=False)

ev.build_supply_table()

ev.supply_plot(ind_axx='vre_scale', ind_plty='C_n')




