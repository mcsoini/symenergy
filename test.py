#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 13:38:17 2018

@author: user
"""

#from sympy import Matrix, S, linsolve, symbols, lambdify, in
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from grimsel.auxiliary.aux_general import print_full

nhours = 2

gamma0, gamma1 = sp.symbols('gamma0, gamma1') # gas variable costsp.
mu0, mu1 = sp.symbols('mu0, mu1', constant=True) # nuclear variable cost
n = sp.symbols('n', positive=True)                     # nuclear production slot 0
g = sp.symbols('g', positive=True)                     # gas production slot 1

c = sp.symbols('c')                     # Charging power
x = sp.symbols('x')                     # Curtailment power
N0 = sp.symbols('N0')                   # legacy nuclear capacity
N = sp.symbols('N', positive=True)                     # Power capacity nuclear
H = sp.symbols('H', positive=True)                     # Power capacity storage

M = sp.symbols('M')                     # O&M fixed cost nuclear
pi = sp.symbols('pi0:%d'%nhours)
pv = sp.symbols('pv0:%d'%nhours, positive=True, constant=True)
eta = sp.symbols('eta', constant=True)  # Round-trip efficiency
L = sp.symbols('L0:%d'%nhours, constant=True)   # Load

lbd_c, lbd_x, lbd_n = sp.symbols('lbd_c, lbd_x, lbd_n') # Shadow price maximum power

lbd_g_p, lbd_n_p = sp.symbols('lbd_g_p, lbd_n_p') # Shadow price postive power


# global list of parameters, not subject to optimization
params = [gamma0, gamma1, mu0, mu1, N0, H, M, *pv, eta, *L]
# global list of variables, subject to optimization
variabs = [g, n, c, N, x]
# global list of lagrange multipliers, subject to optimization
multips = [lbd_g_p, lbd_n_p, lbd_c, lbd_x, lbd_n] + list(pi)

#####################
params = [gamma0, gamma1, mu0, mu1, *pv, *L, N]
params_values = [0.5, 0.04, 0.2, 0.01, 2, 0, 5, 4, 2.5]
params_dict = dict(zip(params, params_values))
variabs = [g, n]
multips = [lbd_g_p, lbd_n_p, lbd_n] + list(pi)
#####################




class cc():
    
    cc_vc_g = (gamma0 + 0.5 * gamma1 * g) * g
    cc_vc_n1 = (mu0 + 0.5 * n * mu1) * n
    cc_vc_n2 = (mu0 + 0.5 * (N0 - N) * mu1) * (N0 - N)
    cc_fc_n = M * (N0 - N)
    const_ld1 = pi[0] * (L[0] - n - pv[0] + c )
    const_ld2 = pi[1] * (L[1] - (N0 - N) - pv[1] - eta * c - g)
    const_c = lbd_c * (c - H)
    const_x = lbd_x * x
    const_n = lbd_n * (n - (N0 - N))

lagrange_0 = (
                cc.cc_vc_g
                + cc.cc_vc_n1 + cc.cc_vc_n2
                + cc.cc_fc_n
                + cc.const_ld1 + cc.const_ld2
               )

class cc():

    cc_vc_g = (gamma0 + 0.5 * gamma1 * g) * g
    cc_vc_n1 = (mu0 + 0.5 * n * mu1) * n
    const_ld1 = pi[0] * (L[0] - n - g - pv[0])
    const_n = lbd_n * (n - N)
    posit_g = lbd_g_p * g
    posit_n = lbd_n_p * n


TC = cc.cc_vc_g + cc.cc_vc_n1 + cc.const_ld1




# Combinations
# c = H
# x = 0
# n = N0 - N



slct_c = [0, 1]
slct_x = [0, 1]
slct_n = [0, 1]
slct_n_p = [0, 1]
slct_g_p = [0, 1]

df_comb = pd.DataFrame(list(itertools.product(slct_c, slct_x, slct_n)),
                       columns=['c', 'x', 'n'])

const_list = ['const_n', 'posit_n', 'posit_g']

df_comb = pd.DataFrame(list(itertools.product(slct_n, slct_n_p, slct_g_p)),
                       columns=const_list)

row = list(df_comb.iterrows())[0]
for row in df_comb.iterrows():
    
    slct_constr = row[1].to_dict()


    lagrange = TC.copy()
    
    
    for constr, is_active in slct_constr.items():
        
        if is_active:
            
            print('%s is active, adding.'%constr)
            lagrange += getattr(cc, constr)

    df_comb.loc[row[0], 'lagrange'] = lagrange


    lagrange_free_symbols = lagrange.free_symbols
    variabs_slct = [ss for ss in lagrange_free_symbols if ss in variabs]
    multips_slct = [ss for ss in lagrange_free_symbols if ss in multips]
    params_slct = [ss for ss in lagrange_free_symbols if ss in params]

    mat = sp.tensor.array.derive_by_array(lagrange, variabs_slct + multips_slct)

    mat = sp.Matrix(mat)
    
    mat = mat.expand()
    
    A, b = sp.linear_eq_to_matrix(mat, variabs_slct + multips_slct)

    result = sp.linsolve((A, b), variabs_slct + multips_slct)

    print(result)

    df_comb.loc[row[0], 'result'] = [result.copy()]

    df_comb.loc[row[0], 'variabs_multips'] =  [[variabs_slct.copy()
                                               + multips_slct.copy()]]


    df_comb.loc[row[0], 'TC'] = TC.copy().subs({var: list(result)[0][ivar]
                                          if not isinstance(result, sp.sets.EmptySet)
                                          else np.nan for ivar, var
                                          in enumerate(variabs_slct + multips_slct)})

    
mask_invalid = (df_comb.result
                       .apply(lambda res: isinstance(res, sp.sets.EmptySet)))   
    
df_comb_invalid = df_comb.loc[mask_invalid]

print('''
The following constraint combinations are invalid:
''', df_comb_invalid[const_list])
                                          

df_comb = df_comb.loc[-mask_invalid]

    

    
# %%


# IDENTIFY LEAST COST SOLUTION

# select which function to plot
func_slct = g

# select independent variable
param_indep = ['N']




# THE FOLLOWING IS APPLIED TO BOTH THE SELECTED FUNCTION AND TC

# get all parameters to be set to a constant value
params_set = [pp for pp in params_slct 
              if not pp.name in param_indep]

# get param values corresponding to params_set in the right order
params_values_set = [params_dict[par] for par in params_set]    

# generate functions only dependent on the indepent variable
for slct_eq_0 in ['TC'] + variabs_slct + multips_slct:
    
    slct_eq = slct_eq_0.name if not isinstance(slct_eq_0, str) else slct_eq_0
#
    

    # function idx depends on constraint, since not all constraints contain the same functions
    if slct_eq != 'TC':

        def get_func_from_idx(x):
            
            mv_list_str = [var.name for var in x.variabs_multips[0]]
            
            if (slct_eq in mv_list_str and
                not isinstance(x.result, sp.sets.EmptySet)):
                
                idx = mv_list_str.index(slct_eq)                
                
                func = list(x.result)[0][idx]
                
                
                return func
            else:
                return np.nan
            
        df_comb[slct_eq] = df_comb.apply(get_func_from_idx, axis=1)




    df_comb[slct_eq + '_lam_res'] = df_comb[slct_eq].apply(lambda func:
                                                       sp.lambdify(params_set, func,
                                                                   modules=['numpy']))

    df_comb[slct_eq + '_res_plot'] = df_comb[slct_eq + '_lam_res'].apply(lambda lam_res:
                                                lam_res(*params_values_set))


    df_comb[slct_eq + '_lam_plot'] = df_comb[slct_eq + '_res_plot'].apply(lambda res_plot: sp.lambdify(param_indep, res_plot, modules=['numpy']))




list_constraints = ['const_n', 'posit_g', 'posit_n']

df_lam_plot = df_comb.set_index(list_constraints).copy()[[c for c in df_comb.columns if '_lam_plot' in c]]

df_lam_plot = df_lam_plot.stack().reset_index().rename(columns={'level_%d'%len(list_constraints): 'func',
                                                  0: 'lambd'})

df_lam_plot = df_lam_plot.set_index(list_constraints + ['func'])


# %%
x_vals = np.linspace(0, 4, 51)

def expand_vals(lam_plot):

    print(lam_plot)
    y_vals = lam_plot.iloc[0](x_vals)
    
    if isinstance(y_vals, float):
        y_vals = np.ones(x_vals.shape) * y_vals
        
    return pd.Series(y_vals, index=pd.Index(x_vals))

# expand all data to selected values
dfg = df_lam_plot.groupby(level=list_constraints + ['func'])['lambd']
df_exp_0 = dfg.apply(expand_vals).reset_index()


df_exp_0 = df_exp_0.rename(columns={'level_%d'%(len(list_constraints) + 1): ':'.join(param_indep)})



# %% EX-POST ENFORCEMENT OF CONSTRAINTS

df_exp = df_exp_0.copy()


is_positive = ['g', 'n']

smaller_n = ['n']


df_exp['is_positive'] = df_exp.func.apply(lambda x: x.split('_')[0]).apply(lambda x: 1 if x in is_positive else 0)
df_exp['smaller_n'] = df_exp.func.apply(lambda x: x.split('_')[0]).apply(lambda x: 1 if x in smaller_n else 0)

print_full(
df_exp.assign(func2=df_exp.func.apply(lambda x: x.split('_')[0])).drop('func', axis=1)
)

mask_valid = pd.Series(1, index=df_exp.index)


df_exp['mv_0'] = mask_valid

# force positive
mask_valid.loc[df_exp.is_positive == 1] *= (df_exp.loc[df_exp.is_positive == 1].lambd >= 0)

df_exp['mv_pos'] = mask_valid

# force greater n
mask_valid.loc[df_exp.smaller_n == 1] *= (df_exp.loc[df_exp.smaller_n == 1].lambd <= df_exp.loc[df_exp.smaller_n == 1].N)


df_exp['mv_n'] = mask_valid

df_exp['mask_valid'] = mask_valid
df_exp.loc[df_exp.const_n.isin([1]) & df_exp.posit_g.isin([0]) & df_exp.posit_n.isin([0])]

#%

mask_valid = df_exp.pivot_table(index=list_constraints + ['N'],
                                values='mask_valid',
                                aggfunc=min)


df_exp = df_exp.drop('mask_valid', axis=1).join(mask_valid, on=mask_valid.index.names)


df_exp.loc[df_exp.mask_valid == 0, 'lambd'] = np.nan

for const in list_constraints:

    df_exp[const] = const + '=' + df_exp[const].astype(str)


df_exp['const_comb'] = df_exp[list_constraints].apply(lambda x: ', '.join(x), axis=1)

# %%
# DEFINE COMBINATION OF CONSTRAINTS

import pyAndy.core.plotpage as pltpg




do = pltpg.PlotPageData.from_df(df=df_exp,
                           ind_axx=['N'], ind_pltx=['func'], ind_plty=[],
                           series=['const_comb'], values=['lambd'],
                           aggfunc=np.mean)


page_kw = dict(left=0.05, right=0.8, bottom=0.1, top=0.95)
pltpg.PlotTiled(do,
                kind_def='plot.line', stacked=False,
                on_values=True,
                marker='.',
                xlabel=param_indep,
                **page_kw)



#y_vals =
##
##y_vals = np.array([np.ones(x_vals.shape) * arr
##                 if not isinstance(arr, np.ndarray)
##                 else arr for arr in y_vals]).T
#
#
#
#fig, ax = plt.subplots(1, 1)
#
#ax.plot(x_vals, y_vals, marker='.')
#ax.set_ylabel(func_slct.name)
#ax.set_xlabel(x_slct.name)
#fig.show()

# %%

y1, y2, x = sp.symbols('y1, y2, x')

equ1 = x**2
equ2 = -x**3 + 2.3
equ3 = x + 4.5

sp.solve(equ1 > equ2)

# %%



from sympy import symbols
from numpy import linspace
from sympy import lambdify
import matplotlib.pyplot as mpl

t = symbols('t')
x = 0.05*t + 0.2/((t - 5)**2 + 2)
lam_x = lambdify(t, x, modules=['numpy'])

x_vals = linspace(0, 10, 100)
y_vals = lam_x(x_vals)

mpl.plot(x_vals, y_vals)
mpl.ylabel("Speed")
mpl.show()

# %%


