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

m = model.Model(curtailment=True)

self = m

m.add_slot(name='day', load=4.5, vre=3)
#m.add_slot(name='night', load=3, vre=1)
m.add_slot(name='evening', load=5, vre=0.5)

#m.add_plant(name='b', vc0=1, vc1=0, slots=m.slots, capacity=1.5)
m.add_plant(name='n', vc0=1, vc1=0, slots=m.slots, capacity=3,
            fcom=10,
            cap_ret=True
            )
m.add_plant(name='g', vc0=2, vc1=0, slots=m.slots)
#
m.add_storage(name='phs',
              eff=0.75,
              slots=m.slots,
              capacity=0.5,
#              energy_capacity=1,
              slots_map={'day': 'chg',
                         'evening': 'dch'
                         })

comp = m.comps['n']
#
m.init_supply_constraints()
m.init_constraint_combinations()

m.remove_mutually_exclusive_combinations()

m.delete_cached()

# %
m.solve_all(nthreads=7)

m.filter_invalid_solutions()


# %%
reload(evaluator)

#m.comps['n'].vc1.value = 0
#m.comps['g'].vc1.value = 0
##m.comps['phs'].C.value = 0.5
#m.slots['day'].vre.value = 0.2
m.slots['evening'].vre.value = 0.5

select_x = m.vre_scale
select_x = m.comps['phs'].C
x_vals = np.linspace(0, 4, 51)
model = m
ev = evaluator.Evaluator(model, select_x, x_vals)

self = ev

list_dep_var = (list(map(str, list(self.model.variabs.keys()) + list(self.model.multips.keys()))))
list_dep_var = ['tc'] + [v for v in list_dep_var if not 'lb_' in v]
ev.get_evaluated_lambdas(list_dep_var)

ev.expand_data_to_x_vals()

ev.df_exp = ev.model.combine_constraint_names(ev.df_exp)

#ev.df_exp.loc[ev.df_exp.const_comb ==
#      ('act_lb_n_pos_p_day=1, '
#      'act_lb_n_pos_p_evening=1, '
#      'act_lb_n_cap_C_day=0, '
#      'act_lb_n_cap_C_evening=0, '
#      'act_lb_g_pos_p_day=1, '
#      'act_lb_g_pos_p_evening=1, '
#      'act_lb_phs_pos_p_day=0, '
#      'act_lb_phs_pos_p_evening=0, '
#      'act_lb_phs_pos_e_None=0, '
#      'act_lb_phs_cap_C_day=0, '
#      'act_lb_phs_cap_C_evening=0, '
#      'act_lb_curt_pos_p_day=0, '
#      'act_lb_curt_pos_p_evening=1')]
# %
ev.enforce_constraints()

ev.init_cost_optimum()

ev.map_func_to_slot()

ev.drop_non_optimal_combinations()

ev.line_plot(all_labels=True)

# %% BUILD SUPPLY CONSTRAINT

df_bal = ev.df_exp.loc[ev.df_exp.const_comb == 'cost_optimum'].copy()

# base dataframe: all operational variables
drop = ['tc', 'pi_', 'lb_']
df_bal = df_bal.loc[-df_bal.func.str.contains('|'.join(drop))]
df_bal.func.unique().tolist()

df_bal = df_bal[['func', 'func_no_slot', 'slot', 'lambd', ev.select_x.name]]

# add parameters
par_add = ['l', 'vre']
pars = [getattr(slot, var) for var in par_add
       for slot in m.slots.values() if hasattr(slot, var)]


df_bal_add = pd.DataFrame(df_bal[ev.select_x.name].drop_duplicates())
for par in pars:
    df_bal_add[par.name] = par.value

df_bal_add = df_bal_add.set_index('C_phs').stack().rename('lambd').reset_index()
df_bal_add = df_bal_add.rename(columns={'level_1': 'func'})
df_bal_add['func_no_slot'] = df_bal_add.func.apply(lambda x: '_'.join(x.split('_')[:-1]))
df_bal_add['slot'] = df_bal_add.func.apply(lambda x: x.split('_')[-1])

df_bal = pd.concat([df_bal, df_bal_add], axis=0, sort=True)

# if ev.select_x == m.scale_vre: join to df_bal and adjust all vre
if ev.select_x == m.vre_scale:
    mask_vre = df_bal.func.str.contains('vre')
    df_bal.loc[mask_vre, 'lambd'] *= df_bal.loc[mask_vre, 'vre_scale']

varpar_neg = ['l', 'curt_p_lam_plot']

df_bal.loc[df_bal.func_no_slot.isin(varpar_neg), 'lambd'] *= -1

varpar_neg = [store.name + '_p_' + slot_name + '_lam_plot'
              for store in m.storages.values()
              for slot_name, chgdch in store.slots_map.items() if chgdch == 'chg']

df_bal.loc[df_bal.func.isin(varpar_neg), 'lambd'] *= -1


data_kw = dict(ind_axx=[select_x.name], ind_pltx=['slot'],
               ind_plty=[], series=['func_no_slot'], values=['lambd'],
               aggfunc=np.mean)
page_kw = dict(left=0.05, right=0.99, bottom=0.050, top=0.8)
plot_kw = dict(kind_def='StackedArea', stacked=True, on_values=True,
               sharex=True, sharey=True, linewidth=4, marker=None,
               xlabel=select_x.name, legend='')

do = pltpg.PlotPageData.from_df(df=df_bal, **data_kw)
plt0 = pltpg.PlotTiled(do, **plot_kw, **page_kw)


lgdplotkey = list(plt0.plotdict.keys())[0]
lgdplot = plt0.plotdict[lgdplotkey]
hdl, lbl = lgdplot.ax.get_legend_handles_labels()

plt0.legend = 'page'
plt0.add_page_legend(lgdplotkey, hdl, lbl)





# %%# adding symbols

select_x_val = 1

tc_subs_vals = m.param_values.copy()

var_vals = ev.df_exp.loc[(ev.df_exp.const_comb == 'cost_optimum')
                & (ev.df_exp.vre_scale == 0.5)]

var_vals = var_vals.set_index('func')['lambd'].to_dict()

list_vars = m.get_variabs_multips_slct(m.tc)

dict_var_symb = {var: [symb for symb in list_vars if symb.name == var.replace('_lam_plot', '')] for var, val in var_vals.items() }
dict_var_symb = {var: symb[0] for var, symb in dict_var_symb.items() if symb}

dict_var_vals = {dict_var_symb[var]: val for var, val in var_vals.items() if var in dict_var_symb.keys()}

tc_subs_vals.update(dict_var_vals)


(m.tc.subs(tc_subs_vals), var_vals['tc_lam_plot'])
