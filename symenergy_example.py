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
m.storages['phs'].C.value = 1
m.storages['phs'].E.value = 1


phs_C_max = m.slots['day'].l.value * 0.5
dd = 20

x_vals = {
         m.vre_scale: np.linspace(0, 0.8, 41),
         m.comps['phs'].C: np.linspace(0, phs_C_max, 2),
         m.comps['phs'].E: np.linspace(0, phs_C_max * dd, 2)
        }

ev = evaluator.Evaluator(m, x_vals)

ev.df_x_vals = ev.df_x_vals.loc[ev.df_x_vals[['C_phs', 'E_phs']].apply(lambda x: tuple(x), axis=1).isin([(0,0), (phs_C_max,phs_C_max * dd)])]

self = ev



list_dep_var = (list(map(str, list(self.model.variabs.keys())
              + list(self.model.multips.keys()))))
list_dep_var = ['tc'] + [v for v in list_dep_var if not 'lb_' in v]
ev.get_evaluated_lambdas(list_dep_var)


# %%
slct_const_comb = \
['act_lb_n_pos_p_day=0, act_lb_n_pos_p_night=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_night=0, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_night=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_night=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_night=0, act_lb_phs_e_cap_E_None=0, act_lb_curt_pos_p_day=1, act_lb_curt_pos_p_night=1',
 'act_lb_n_pos_p_day=0, act_lb_n_pos_p_night=0, act_lb_n_pos_C_ret_None=0, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_night=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_night=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_night=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_night=0, act_lb_phs_e_cap_E_None=0, act_lb_curt_pos_p_day=1, act_lb_curt_pos_p_night=1']

# %%


ev.df_lam_plot = ev.df_lam_plot.reset_index().loc[ev.df_lam_plot.reset_index().const_comb.isin(slct_const_comb)].set_index(ev.df_lam_plot.index.names)


ev.expand_to_x_vals()

ev.df_exp = ev.df_exp.loc[ev.df_exp.vre_scale == 0.3]

ev.enforce_constraints()

ev.init_cost_optimum()

ev.map_func_to_slot()

#ev.drop_non_optimal_combinations()

ev.build_supply_table()


# %%

ev.line_plot(all_labels=False)

ev.supply_plot(ind_axx=['vre_scale'], ind_plty=['C_phs'])



# %%

ind_axx=['vre_scale']
ind_plty=['C_phs']



data_kw = dict(ind_axx=ind_axx, ind_pltx=['slot'],
               ind_plty=ind_plty,
               series=['func_no_slot'],
               values=['lambd'],
               aggfunc=np.mean)
page_kw = dict(left=0.05, right=0.99, bottom=0.050, top=0.8)
plot_kw = dict(kind_def='StackedArea', stacked=True, on_values=True,
               sharex=True, sharey=True, linewidth=4, marker='o',
               xlabel=ind_axx, legend='')

do = pltpg.PlotPageData.from_df(df=ev.df_bal, **data_kw)
plt0 = pltpg.PlotTiled(do, **plot_kw, **page_kw)


lgdplotkey = list(plt0.plotdict.keys())[0]
lgdplot = plt0.plotdict[lgdplotkey]
hdl, lbl = lgdplot.ax.get_legend_handles_labels()

plt0.legend = 'page'
plt0.add_page_legend(lgdplotkey, hdl, lbl)

lgdplotkey = list(plt0.plotdict.keys())[-1]
plt0.plotdict[lgdplotkey].ax.legend()





ev.df_exp.loc[ev.df_exp.is_optimum & (ev.df_exp.C_phs > 0)].pivot_table(values='vre_scale',
                                                index='const_comb',
                                                aggfunc=[min, max]).T



ev.const_comb_opt

# %% REPORT CONSTRAINT COMBINATIONS

dict_const_combs = {
'Peak replacement day': 'act_lb_n_pos_p_day=0, act_lb_n_pos_p_night=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_night=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=0, act_lb_g_pos_p_night=0, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_night=1, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_night=0, act_lb_phs_e_cap_E_None=0, act_lb_curt_pos_p_day=1, act_lb_curt_pos_p_night=1',
'Storage peak replacement night': 'act_lb_n_pos_p_day=0, act_lb_n_pos_p_night=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_night=1, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_night=0, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_night=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_night=0, act_lb_phs_e_cap_E_None=0, act_lb_curt_pos_p_day=1, act_lb_curt_pos_p_night=1',
'Storage base load replacement night': 'act_lb_n_pos_p_day=0, act_lb_n_pos_p_night=0, act_lb_n_pos_C_ret_None=1, act_lb_n_p_cap_C_day=1, act_lb_n_p_cap_C_night=0, act_lb_n_C_ret_cap_C_None=0, act_lb_g_pos_p_day=1, act_lb_g_pos_p_night=1, act_lb_phs_pos_p_day=0, act_lb_phs_pos_p_night=0, act_lb_phs_pos_e_None=0, act_lb_phs_p_cap_C_day=0, act_lb_phs_p_cap_C_night=0, act_lb_phs_e_cap_E_None=0, act_lb_curt_pos_p_day=1, act_lb_curt_pos_p_night=1'
}



dict_constrs = {
'act_lb_n_pos_p_day': 'Base load production zero day',
'act_lb_n_pos_p_night': 'Base load production zero night',
'act_lb_n_pos_C_ret_None': 'Zero retirement',
'act_lb_n_p_cap_C_day': 'Maximum base load production day',
'act_lb_n_p_cap_C_night': 'Maximum base load production night',
'act_lb_n_C_ret_cap_C_None': 'Maximum retirement',
'act_lb_g_pos_p_day': 'Peak load production zero day',
'act_lb_g_pos_p_night': 'Peak load production zero night',
'act_lb_phs_pos_p_day': 'Zero storage charging day',
'act_lb_phs_pos_p_night': 'Zero storage charging night',
'act_lb_phs_pos_e_None': 'Zero stored energy',
'act_lb_phs_p_cap_C_day': 'Maximum storage charging day',
'act_lb_phs_p_cap_C_night': 'Maximum storage charging night',
'act_lb_phs_e_cap_E_None': 'Maximum charged energy',
'act_lb_curt_pos_p_day': 'Zero curtailment day',
'act_lb_curt_pos_p_night': 'Zero curtailment night'}

table_const_combs = m.df_comb.loc[m.df_comb.const_comb.isin(ev.const_comb_opt), m.constrs_cols_neq + ['const_comb']]

table_const_combs.columns = [dict_constrs[cstr] for cstr in table_const_combs.columns if not cstr == 'const_comb'] + ['const_comb']
table_const_combs['const_comb'] = table_const_combs.const_comb.replace({vv: kk for kk, vv in dict_const_combs.items()})

table_const_combs.T







