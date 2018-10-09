#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Evaluator class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import sympy as sp
import numpy as np
import pandas as pd

import symenergy.evaluator.plotting as plotting

class Evaluator(plotting.EvPlotting):
    '''
    Evaluates model results for selected
    '''

    def __init__(self, model, select_x, x_vals):
        '''
        Keyword arguments:
            * model -- symenergy model
            * select_x -- symenergy Parameter; to be varied according to x_vals
            * x_vals -- iterable with value for the evaluation of select_x
        '''

        self.model = model
        self.select_x = select_x
        self.x_vals = x_vals

        self.dfev = model.df_comb.copy()

        self.model.init_total_param_values()

        print('param_values=', self.model.param_values)

    def get_evaluated_lambdas(self, list_dep_var=None):
        '''
        For each dependent variable and total cost get a lambda function
        evaluated by constant parameters. This subsequently evaluated
        for all x_pos.

        Generated attributes:
            - df_lam_plot: Holds all lambda functions for each dependent variable and each constraint combination.
        '''

        # get dependent variables (variabs and multips)
        if not list_dep_var:

          list_dep_var = ['tc'] + (list(map(str, list(self.model.variabs.keys())
                                                 + list(self.model.multips.keys()))))

        slct_eq_0 = ('n_p_day' if 'n_p_day' in list_dep_var
                     else list_dep_var[0])
        for slct_eq_0 in list_dep_var:

            print('Extracting solution for %s'%slct_eq_0)

            slct_eq = (slct_eq_0.name
                       if not isinstance(slct_eq_0, str)
                       else slct_eq_0)

            # function idx depends on constraint, since not all constraints
            # contain the same functions
            if slct_eq != 'tc':

                x = self.dfev.iloc[0]

                self.dfev[slct_eq] = self.dfev.apply(lambda x: self._get_func_from_idx(x, slct_eq), axis=1)

            x = self.dfev[slct_eq].iloc[0]
            self.dfev['%s_expr_plot'%slct_eq] = self.dfev[slct_eq].apply(self._subs_param_values)

            self.dfev[slct_eq + '_lam_plot'] = self.dfev[slct_eq + '_expr_plot'].apply(lambda res_plot: sp.lambdify(self.select_x.symb, res_plot, modules=['numpy']))

        df_lam_plot = self.dfev.set_index(list(map(str, self.model.constrs_cols_neq)) + ['const_comb']).copy()[[c for c in self.dfev.columns if isinstance(c, str) and '_lam_plot' in c]]

        col_names = {'level_%d'%(len(self.model.constrs_cols_neq) + 1): 'func',
                     0: 'lambd'}
        df_lam_plot = (df_lam_plot.stack().reset_index()
                                  .rename(columns=col_names))
        df_lam_plot = df_lam_plot.set_index(self.model.constrs_cols_neq + ['const_comb', 'func'])

        self.df_lam_plot = df_lam_plot

    def _get_func_from_idx(self, x, slct_eq):
        '''
        Get result expression corresponding to the selected variable slct_eq.

        From the result set of row x, get the expression corresponding to the
        selected variable/multiplier slct_eq. This first finds the index
        of the corresponding expression through comparison with slct_eq and
        then returns the expression itself.
        '''

        mv_list_str = [var.name for var in x.variabs_multips]

        if (slct_eq in mv_list_str and
            not isinstance(x.result, sp.sets.EmptySet)):

            idx = mv_list_str.index(slct_eq)

            func = list(x.result)[0][idx]

            return func
        else:
            return np.nan

    def _subs_param_values(self, x):
        '''
        Substitutes all parameter values except for
        the one selected as single independent variable.
        '''

        if isinstance(x, float) and np.isnan(x):
            return np.nan
        else:
            x_ret = x.subs({kk: vv for kk, vv
                            in self.model.param_values.items()
                            if not kk == self.select_x.symb})
            return x_ret

    def get_expanded_row(self, lam_plot):
        '''
        Apply single lambda function/row to the self.x_vals.

        Input parameters:
            * lam_plot -- Single row of the self.df_lam_plot dataframe.

        Return values:
            * Series with y values
        '''

        y_vals = lam_plot.iloc[0](self.x_vals)

        if isinstance(y_vals, float):
            y_vals = np.ones(self.x_vals.shape) * y_vals

        return pd.Series(y_vals, index=pd.Index(self.x_vals))

    def expand_data_to_x_vals(self):

        # expand all data to selected values
        group_levels = self.model.constrs_cols_neq + ['const_comb', 'func']
        dfg = self.df_lam_plot.groupby(level=group_levels)['lambd']



        df_exp_0 = dfg.apply(self.get_expanded_row).reset_index()

        col_names = {'level_%d'%(len(self.model.constrs_neq) + 2):
                     self.select_x.name}
        df_exp_0 = df_exp_0.rename(columns=col_names)

        self.df_exp = df_exp_0

    def _init_constraints_active(self):
        '''
        Create binary columns depending on whether the plant constraints
        are active or not.
        '''

        # the following lists are attributes just so they can be getattr'd
        self.is_positive = self.model.get_all_is_positive()
        self.is_capacity_constrained = \
                self.model.get_all_is_capacity_constrained()

        get_var = lambda x: x.split('_lam_')[0]
        set_constr = lambda x, lst: 1 if x in map(str, getattr(self, lst)) else 0

        lst = 'is_positive'
        for lst in ['is_positive', 'is_capacity_constrained']:

#            map_constr = {name: name in map(str, getattr(self, lst))
#                   for name in self.df_exp.func.apply(get_var).unique().tolist()}

            constr_act = (self.df_exp.func.apply(get_var)
                                          .apply(lambda x: set_constr(x, lst)))
            self.df_exp[lst] = constr_act

            self.df_exp[[lst, 'func']].drop_duplicates()



    def _get_mask_valid_solutions(self):

        mask_valid = pd.Series(1, index=self.df_exp.index)

        self.df_exp['mv_0'] = mask_valid.copy()

        # filter positive
        msk_pos = self.df_exp.is_positive == 1
        constraint_met = self.df_exp.loc[msk_pos].lambd >= 0
        mask_valid.loc[msk_pos] *= constraint_met

        self.df_exp['mv_pos'] = mask_valid.copy()

        dict_cap = {cap: val
                    for comp in self.model.comps.values()
                    if hasattr(comp, 'get_constrained_variabs') # exclude slots
                    for cap, val in comp.get_constrained_variabs().items()}

        if dict_cap:
            msk_cap_cstr = self.df_exp.is_capacity_constrained == 1

#            C, pp = list(dict_cap.items())[0]
            for C, pp in dict_cap.items():

                slct_func = ['%s_lam_plot'%symb.name for symb in pp]

                # things are different depending on whether or not select_x is the corresponding capacity
                if self.select_x is C:
                    val_cap = self.df_exp.loc[msk_cap_cstr, self.select_x.name]
                else:
                    val_cap = C.value

                # need to add retired and additional capacity
                for addret, sign in {'add': +1, 'ret': -1}.items():
                    if hasattr(pp, 'C_%s'%addret):
                        name_func = '_'.join(list(reversed(C.name.split('_'))) + [addret, 'None'])
                        df_C = self.df_exp.loc[self.df_exp.func.str.contains(name_func)].copy()
                        df_C = df_C.set_index(['const_comb', self.select_x.name])['lambd'].rename('_C_%s'%addret)
                        self.df_exp = self.df_exp.join(df_C, on=df_C.index.names)
                        val_cap = val_cap + sign * self.df_exp.loc[msk_cap_cstr, '_C_%s'%addret]

                mask_slct_func = self.df_exp.func.isin(slct_func)
                constraint_met = self.df_exp.lambd.copy()
                constraint_met.loc[:] = True
                constraint_met.loc[mask_slct_func] = \
                    self.df_exp.loc[mask_slct_func].lambd <= val_cap

                # delete temporary columns:
                self.df_exp = self.df_exp[[c for c in self.df_exp.columns
                                        if not c in ['_C_ret', '_C_add']]]

                mask_valid.loc[msk_cap_cstr] *= constraint_met

            self.df_exp['mv_n'] = mask_valid.copy()

        self.df_exp['mask_valid'] = mask_valid.copy()

        # consolidate mask by constraint combination and x values
        index = self.model.constrs_cols_neq + [self.select_x.name, 'const_comb']
        mask_valid = self.df_exp.pivot_table(index=index,
                                             values='mask_valid',
                                             aggfunc=min)

        self.df_exp.drop('mask_valid', axis=1, inplace=True)

        return mask_valid
# %%
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
# %%


    def enforce_constraints(self):
        '''
        Discard solutions which violate any of the
            * positive
            * capacity
        constraints.
        TODO: Ideally this would be modular and part of the components.
        '''

        self._init_constraints_active()

        mask_valid = self._get_mask_valid_solutions()

        self.df_exp = self.df_exp.join(mask_valid, on=mask_valid.index.names)

        self.df_exp.loc[self.df_exp.mask_valid == 0, 'lambd'] = np.nan


    def init_cost_optimum(self):
        ''' Adds cost optimum column to the expanded dataframe. '''

        tc = self.df_exp.loc[self.df_exp.func == 'tc_lam_plot'].copy()

        tc_min = (tc.groupby(self.select_x.name, as_index=0)
                    .apply(lambda x: x.nsmallest(1, 'lambd')))

        self.const_comb_opt = tc_min.const_comb.unique().tolist()

        merge_on = [self.select_x.name] + self.model.constrs_cols_neq
        df_exp_min = pd.merge(tc_min[merge_on], self.df_exp,
                              on=merge_on, how='inner')


        df_exp_min['const_comb'] = 'cost_optimum'

        self.df_exp = pd.concat([self.df_exp, df_exp_min],
                                axis=0, sort=False)
        self.df_exp = self.df_exp.reset_index(drop=True)

    def drop_non_optimal_combinations(self):


        constrs_opt = self.df_exp.loc[self.df_exp.const_comb == 'cost_optimum']
        constrs_opt = constrs_opt[self.model.constrs_cols_neq].drop_duplicates()
        constrs_opt = self.model.combine_constraint_names(constrs_opt)['const_comb'].tolist()
        constrs_opt += ['cost_optimum']

        self.df_exp = self.df_exp.loc[self.df_exp.const_comb.isin(constrs_opt)]

    def map_func_to_slot(self):

        func_list = self.df_exp.func.unique()

        slot_name_list = list(self.model.slots.keys())

        func_map = {func: '+'.join([ss for ss in slot_name_list
                                    if ss in func])
                    for func in func_list}

        func_map = {func: slot if not slot == '' else 'global'
                    for func, slot in func_map.items()}

        self.df_exp['slot'] = self.df_exp['func'].replace(func_map)

        del_slot = lambda x: x.func.replace('_%s'%x.slot, '')

        self.df_exp['func_no_slot'] = (self.df_exp[['slot', 'func']]
                                           .apply(del_slot, axis=1))
