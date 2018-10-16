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

m = model.Model(curtailment=True, nthreads=7)

self = m

m.add_slot(name='day', load=4.5, vre=3)
m.add_slot(name='evening', load=5, vre=0.5)

m.add_plant(name='n', vc0=1, vc1=0, slots=m.slots, capacity=3,
            fcom=10,
            cap_ret=True
            )
m.add_plant(name='g', vc0=2, vc1=0, slots=m.slots)

m.add_storage(name='phs',
              eff=0.75,
              slots=m.slots,
              capacity=0.5,
#              energy_capacity=1,
              slots_map={'day': 'chg',
                         'evening': 'dch'
                         })

m.generate_solve()


# fix stored energy
store = m.storages['phs']

def get_result_dict(x, string_keys=False):

    dict_res = {str(var): res for var, res in zip(x.variabs_multips, x.result[0])}
    return dict_res

def set_phs_p_day_zero(x, eff):

    dict_var = get_result_dict(x, True)

    dict_var['phs_e_None'] = dict_var['phs_p_day'] * eff**0.5

    return [[dict_var[str(var)] for var in x.variabs_multips]]


m.df_comb['result'] = m.df_comb.apply(set_phs_p_day_zero, args=(store.eff.symb,), axis=1)


# %%
reload(evaluator)

m.comps['n'].vc1.value = 0
m.comps['n'].vc0.value = 1.9 / 0.33 + 3.9
m.comps['n'].fcom.value = 200 #/ 8760 * 2
m.comps['n'].C.value = 4000

m.comps['g'].vc1.value = 0
m.comps['g'].vc0.value = 29.5 / 0.21 + 20

m.slots['day'].vre.value = (6569 + 5239)
m.slots['evening'].vre.value = 0
m.slots['day'].l.value = 6569
m.slots['evening'].l.value = 5239


m.storages['phs'].eff.value = 0.75
m.storages['phs'].C.value = 2000
#m.storages['phs'].E.value = 20000

m.vre_scale.value = 0.5

x_vals = {m.vre_scale: np.linspace(0, 1, 21),
          m.comps['n'].C: np.linspace(0, 3285, 3)
         }

m.init_total_param_values()


#m.df_comb = m.df_comb.loc[m.df_comb.const_comb == dict_const_comb_confl['expected']]

model = m
ev = evaluator.Evaluator(model, x_vals)

self = ev

list_dep_var = (list(map(str, list(self.model.variabs.keys()) + list(self.model.multips.keys()))))
list_dep_var = ['tc'] + [v for v in list_dep_var if not 'lb_' in v]
ev.get_evaluated_lambdas(list_dep_var)







# %%

# optimum original
['act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=0, act_lb_g_pos_p_evening=0, act_lb_phs_pos_p_day=1, act_lb_phs_pos_p_evening=1, act_lb_phs_pos_e_None=1, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
 'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=0, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=0, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=1, act_lb_phs_p_cap_C_evening=0']
# optimum force
const_comb_force = \
['act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=0, act_lb_g_pos_p_evening=0, act_lb_phs_pos_p_day=1, act_lb_phs_pos_p_evening=1, act_lb_phs_pos_e_None=1, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=0, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=0, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=1, act_lb_phs_p_cap_C_evening=0']

# optimum conflict

const_comb_confl = \
['act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=0, act_lb_g_pos_p_evening=0, act_lb_phs_pos_p_day=1, act_lb_phs_pos_p_evening=1, act_lb_phs_pos_e_None=1, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=0, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=0, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=0, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=1, act_lb_phs_p_cap_C_evening=0',
'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=0, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=1, act_lb_phs_p_cap_C_evening=0',
'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0']

dict_const_comb_confl = {'rep peak': 'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=0, act_lb_g_pos_p_evening=0, act_lb_phs_pos_p_day=1, act_lb_phs_pos_p_evening=1, act_lb_phs_pos_e_None=1, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'rep. pk storage': 'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=0, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'weird':           'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=0, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
'rep bs storage':  'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=0, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=1, act_lb_phs_p_cap_C_evening=0',
'storage constr':  'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=0, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=1, act_lb_phs_p_cap_C_evening=0',
'expected':        'act_lb_n_pos_p_day=0, act_lb_n_pos_p_evening=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_evening=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_evening=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_evening=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_evening=0',
}


# %%

# phs_e_None_lam_plot of expected is empty!

res = m.df_comb.loc[m.df_comb.const_comb == (dict_const_comb_confl['expected'])].result.values[0]

var = m.df_comb.loc[m.df_comb.const_comb == (dict_const_comb_confl['expected'])].variabs_multips.values[0]

dict_res = {str(v): r for v, r in zip(var, res[0])}

phs = m.comps['phs']

sp.simplify('lb_phs_pwrerg_None')


# %%

ev.df_lam_plot = ev.df_lam_plot.loc[ev.df_lam_plot.index.get_level_values('const_comb').str.contains('|'.join(const_comb_confl))]

ev.x_vals = {m.vre_scale: np.linspace(0, 0.8, 81),
             m.comps['n'].C: np.linspace(4000, 4000, 1)
            }


ev.expand_to_x_vals()

#ev.df_exp = ev.df_exp.loc[ev.df_exp.vre_scale.isin(np.arange(0.385, 0.4, 0.005))]

#ev.df_exp['const_comb'] = ev.df_exp.const_comb.replace({vv: kk for kk, vv in dict_const_comb_confl.items()})
#ev.df_exp = ev.df_exp.loc[ev.df_exp.const_comb.isin(['rep bs storage', 'weird'])]


ev.enforce_constraints()

ev.init_cost_optimum()

ev.map_func_to_slot()

ev.drop_non_optimal_combinations()

ev.build_supply_table()

#ev.df_exp.loc[-ev.df_exp.lambd.isnull()]
# %%
print_full(
ev.df_exp.loc[
#        (ev.df_exp.const_comb.isin(const_comb_confl[0:3]))
#             (ev.df_exp.vre_scale == 0.405)
           (-ev.df_exp.lambd.isnull())
             ]#.drop('const_comb', axis=1)
.pivot_table(index=['func'], columns=['const_comb', 'vre_scale'], values='lambd'))
# %%


ev.line_plot(all_labels=False)

#ev.df_bal = ev.df_bal.loc[ev.df_bal.fcom_n == 200]

ev.supply_plot(ind_axx=['vre_scale'], ind_plty=[])




