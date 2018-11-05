#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Evaluator class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""

import sympy as sp
import numpy as np
import pandas as pd
import itertools
import time
import multiprocessing

import symenergy.evaluator.plotting as plotting
import grimsel.auxiliary.aux_sql_func as aql

from symenergy.auxiliary.parallelization import parallelize_df

class Evaluator(plotting.EvPlotting):
    '''
    Evaluates model results for selected
    '''

    def __init__(self, model, x_vals, drop_non_optimum=True,
                 eval_accuracy=1e-9, nthreads=None, to_sql=False):
        '''
        Keyword arguments:
            * model -- symenergy model
           ( * select_x -- symenergy Parameter; to be varied
                           according to x_vals )
            * x_vals -- iterable with value for the evaluation of select_x
            * eval_accuracy -- absolute slack for constraint evaluation
        '''

        self.model = model

        self.nthreads = nthreads
        self.to_sql = to_sql

        self.drop_non_optimum = drop_non_optimum

        self.x_vals = x_vals

        self.eval_accuracy = eval_accuracy

        self.dfev = model.df_comb.copy()

        self.model.init_total_param_values()
        self.is_positive = self.model.get_all_is_positive()

        self.df_x_vals = self.get_x_vals_combs()

        print('param_values=', self.model.param_values)


    @property
    def x_vals(self):
        return self._x_vals

    @x_vals.setter
    def x_vals(self, x_vals):
        self._x_vals = x_vals#self.sanitize_x_vals(x_vals)
        self.x_symb = [x.symb for x in self._x_vals.keys()]
        self.x_name = [x.name for x in self.x_symb]

        self.df_x_vals = self.get_x_vals_combs()

    def get_x_val_steps(self):
        '''
        Assumes constant step size.
        '''

        return {par.name: (val[1] - val[0]) if len(val) > 1 else None
                for par, val in self.x_vals.items()}

    def sanitize_x_vals(self, x_vals):
        '''
        TODO: Figure out why this is necessary for the numerical evaluation
        of the results!
        '''

        return {comp: [int(val * 100) / 100 for val in vals]
                for comp, vals in x_vals.items()}


    def _get_list_dep_var(self, skip_multipliers=False):

        list_dep_var = list(map(str, list(self.model.variabs.keys())
                                   + list(self.model.multips.keys())))
        list_dep_var += ['tc']

        if skip_multipliers:
            list_dep_var = [v for v in list_dep_var if not 'lb_' in v]

        return list_dep_var

    def get_evaluated_lambdas(self, skip_multipliers=False):
        '''
        For each dependent variable and total cost get a lambda function
        evaluated by constant parameters. This subsequently evaluated
        for all x_pos.

        Generated attributes:
            - df_lam_plot: Holds all lambda functions for each dependent
                           variable and each constraint combination.
        '''

        # get dependent variables (variabs and multips)
        list_dep_var = self._get_list_dep_var(skip_multipliers)

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

                get_func = lambda x: self._get_func_from_idx(x, slct_eq)
                self.dfev[slct_eq] = self.dfev.apply(get_func, axis=1)

            self.dfev['%s_expr_plot'%slct_eq] = \
                        self.dfev[slct_eq].apply(self._subs_param_values)

            lambdify = lambda res_plot: sp.lambdify(self.x_symb, res_plot,
                                                    modules=['numpy'])

            self.dfev[slct_eq + '_lam_plot'] = (
                            self.dfev['%s_expr_plot'%slct_eq].apply(lambdify))

        idx = list(map(str, self.model.constrs_cols_neq)) + ['const_comb']
        cols = [c for c in self.dfev.columns
                if isinstance(c, str)
                and '_lam_plot' in c]
        df_lam_plot = self.dfev.set_index(idx).copy()[cols]


        col_names = {'level_%d'%(len(self.model.constrs_cols_neq) + 1): 'func',
                     0: 'lambd_func'}
        df_lam_plot = (df_lam_plot.stack().reset_index()
                                  .rename(columns=col_names))
        df_lam_plot = (df_lam_plot.reset_index(drop=True)
                                  .reset_index()
                                  .rename(columns={'index': 'idx'}))
        df_lam_plot = df_lam_plot.set_index(self.model.constrs_cols_neq
                                            + ['const_comb', 'func', 'idx'])

        self.df_lam_plot = df_lam_plot



    def get_x_vals_combs(self):
        '''
        Generates dataframe with all combinations of x_vals.

        Used as default by expand_to_x_vals or can be used externally to
        select subsets of
        '''

        return pd.DataFrame(list(itertools.product(*self.x_vals.values())),
                            columns=[col.name for col in self.x_vals.keys()])


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
                            if not kk in [select_x.symb
                                          for select_x
                                          in self.x_vals.keys()]})
            return x_ret


    def _get_expanded_row(self, lam_plot, x_vals):
        '''
        Apply single lambda function/row to the self.x_vals.

        Input parameters:
            * lam_plot -- Single row of the self.df_lam_plot dataframe.

        Return values:
            * Series with y values
        '''

        y_vals = [lam_plot.iloc[0](*val_row) for val_row in x_vals]

        if isinstance(y_vals, float):
            y_vals = np.ones(len(x_vals)) * y_vals

        return pd.Series(y_vals, index=pd.Index(x_vals))

    def _evaluate(self, df):
        '''
        Input dataframe:
            df[['func', 'const_comb', 'lambd', 'self.x_name[0]', ...]]
        Returns expanded data for all rows in the input dataframe.
        '''

        if __name__ == '__main__':
            x = df[['func', 'const_comb', 'lambd_func'] + self.x_name].iloc[0]

        import time

        tm = {}
        tm['t'] = time.time()

        def report(tok, index):
            if index % 10000 == 0:

                print(index / len(df) * 100, time.time() - tm['t'])
                tm['t'] = time.time()

        def process(x, report=None):
            return x.lambd_func(*x.loc[self.x_name])

        return df.apply(process, axis=1)

#        return df.apply(process, args=(report,), axis=1)


    def init_schema(self):

        cols = [('func', 'VARCHAR'),
                ('const_comb', 'VARCHAR'),
                ('is_positive', 'SMALLINT'),
                ('lambd', 'DOUBLE PRECISION'),
                ('mask_valid', 'BOOLEAN'),
                ('is_optimum', 'BOOLEAN'),
                ] + [(x, 'DOUBLE PRECISION') for x in self.x_name]

        aql.init_table('test_evaluator', cols, db='storage2')

    def evaluate_by_x(self, x, df):
        print(x)

        # add x val columns
        for col in self.x_name:
            df[col] = x[col]

        df['lambd'] = self._evaluate(df)

        mask_valid = self._get_mask_valid_solutions(df)
        df = df.join(mask_valid, on=mask_valid.index.names)
        df.loc[df.mask_valid == 0, 'lambd'] = np.nan

        df['is_optimum'] = self.init_cost_optimum(df)

        if self.drop_non_optimum == True:
            df = df.loc[df.is_optimum]

        print(time.time() - t)

        if self.to_sql:
            aql.write_sql(df[['func',
                             'const_comb',
                             'is_positive',
                             'lambd',
                             'mask_valid',
                             'is_optimum',
                             'vre_scale',
                             'vc1_g'] + self.x_name], 'storage2', sc='public', tb='test_evaluator', if_exists='append')
        else:
            return df

    def evaluate_all(self):

        df_lam_plot = self.df_lam_plot.reset_index()[['func',
                                                      'const_comb',
                                                      'lambd_func']]

        df = pd.merge(self.df_x_vals.assign(key=1),
                      df_lam_plot.assign(key=1), on='key')

        df['lambd'] = self._evaluate(df)


    def call_evaluate_by_x(self, df_x, df_lam):

        eval_x = lambda x: self.evaluate_by_x(x, df_lam)

        if __name__ == '__main__':
            x = df_x.iloc[0]
            df = df_lam
            eval_x(x)

        result = self.df_x_vals.apply(eval_x, axis=1)


        return result

    def expand_to_x_vals(self):
        '''
        Applies evaluate_by_x to all df_x_vals rows.
            * by_x_vals -- if True: expand x_vals for all const_combs/func
                           if False: expand const_combs/func for all x_vals
        '''

        df_lam_plot = self.df_lam_plot.reset_index()[['func',
                                                      'const_comb',
                                                      'lambd_func']]

        df_lam_plot = self._init_constraints_active(df_lam_plot)

        dfg = self.df_x_vals.groupby(level=0, as_index=False)

#        if __name__ == '__main__':
#            x = dfg.get_group(list(dfg.groups.keys())[0])
#            df = df_lam_plot.copy()
#

        df_x = self.df_x_vals
        df_lam = df_lam_plot
        if not self.nthreads:
            df_exp_0 = self.call_evaluate_by_x(df_x, df_lam)
        else:
            func = self.call_evaluate_by_x
            nthreads = self.nthreads
            df_exp_0 = parallelize_df(df_x, func, nthreads, df_lam=df_lam)




        if not self.to_sql:
            df_exp_0 = pd.concat(df_exp_0.tolist())

            df_exp_0 = df_exp_0.reset_index(drop=True)

            self.df_exp = df_exp_0

            self.const_comb_opt = (self.df_exp.loc[self.df_exp.is_optimum,
                                                   'const_comb'].unique().tolist())



    def _init_constraints_active(self, df):
        '''
        Create binary columns depending on whether the plant constraints
        are active or not.
        '''

        # the following lists are attributes just so they can be getattr'd

        get_var = lambda x: x.split('_lam_')[0]
        set_constr = lambda x, lst: (1 if x in map(str, getattr(self, lst))
                                     else 0)

        lst = 'is_positive'
        for lst in ['is_positive']:

            constr_act = (df.func.apply(get_var)
                                 .apply(lambda x: set_constr(x, lst)))
            df[lst] = constr_act

            df[[lst, 'func']].drop_duplicates()

        return df


    def _get_mask_valid_positive(self, df):

        msk_pos = df.is_positive == 1
        mask_positive = pd.Series(True, index=df.index)
        mask_positive.loc[msk_pos] = df.loc[msk_pos].lambd + self.eval_accuracy >= 0

        return mask_positive

    def _get_mask_valid_capacity(self, df):

        dict_cap = [(cap, val)
                    for comp in self.model.comps.values()
                    if hasattr(comp, 'get_constrained_variabs') # exclude slots
                    for cap, val in comp.get_constrained_variabs()]

        mask_valid = pd.Series(True, index=df.index)

        if dict_cap:


#            C, pp = dict_cap[0]
            for C, pp in dict_cap:

                print('Valid capacity constraint %s, %s'%(C.name, pp))

                slct_func = ['%s_lam_plot'%symb.name for symb in pp]

                mask_slct_func = df.func.isin(slct_func)

                # things are different depending on whether or not select_x is the corresponding capacity
                if C in self.x_vals.keys():
                    val_cap = df[C.name]
                else:
                    val_cap = pd.Series(C.value, index=df.index)

                # need to add retired and additional capacity
                for addret, sign in {'add': +1, 'ret': -1}.items():
                    func_C_addret = [variab for variab in slct_func if 'C_%s_None'%addret in variab]
                    func_C_addret = func_C_addret[0] if func_C_addret else None
                    if func_C_addret:
                        mask_addret = (df.func.str
                                                  .contains(func_C_addret))
                        df_C = df.loc[mask_addret].copy()
                        df_C = df_C.set_index(['const_comb'] + self.x_name)['lambd'].rename('_C_%s'%addret)
                        df = df.join(df_C, on=df_C.index.names)

                        # doesn't apply to itself, hence -mask_addret
                        val_cap.loc[-mask_addret] += \
                            + sign * df.loc[-mask_addret,
                                                     '_C_%s'%addret]

                constraint_met = pd.Series(True, index=df.index)
                constraint_met.loc[mask_slct_func] = \
                                    (df.loc[mask_slct_func].lambd * (1 - self.eval_accuracy)
                                     <= val_cap.loc[mask_slct_func])

#                # delete temporary columns:
#                df = df[[c for c in df.columns
#                                        if not c in ['_C_ret', '_C_add']]]

                mask_valid &= constraint_met

#            self.df_exp['mv_n'] = mask_valid.copy()

        return mask_valid

    def _get_mask_valid_solutions(self, df):

        mask_valid = pd.Series(True, index=df.index)

        mask_positive = self._get_mask_valid_positive(df)
        mask_valid = mask_valid & mask_positive

        mask_capacity = self._get_mask_valid_capacity(df)
        mask_valid &= mask_capacity

        df['mask_valid'] = mask_valid

        # consolidate mask by constraint combination and x values
        index = self.x_name + ['const_comb']
        mask_valid = df.pivot_table(index=index,
                                    values='mask_valid',
                                    aggfunc=min)

        df.drop('mask_valid', axis=1, inplace=True)

        return mask_valid


    def build_supply_table(self, df=None):
        '''
        Generates a table representing the supply constraint for easy plotting.
        '''

        if not isinstance(df, pd.DataFrame):
            df=self.df_exp

        df_bal = df.loc[df.is_optimum].copy()

        # base dataframe: all operational variables
        drop = ['tc', 'pi_', 'lb_']
        df_bal = df_bal.loc[-df_bal.func.str.contains('|'.join(drop))]

        df_bal = df_bal[['func', 'const_comb', 'func_no_slot',
                         'slot', 'lambd'] + self.x_name]

        # add parameters
        par_add = ['l', 'vre']
        pars = [getattr(slot, var) for var in par_add
               for slot in self.model.slots.values() if hasattr(slot, var)]

        df_bal_add = pd.DataFrame(df_bal[self.x_name].drop_duplicates())
        for par in pars:
            df_bal_add[par.name] = par.value

        df_bal_add = df_bal_add.set_index(self.x_name).stack().rename('lambd').reset_index()
        df_bal_add = df_bal_add.rename(columns={'level_%d'%len(self.x_name): 'func'})
        df_bal_add['func_no_slot'] = df_bal_add.func.apply(lambda x: '_'.join(x.split('_')[:-1]))
        df_bal_add['slot'] = df_bal_add.func.apply(lambda x: x.split('_')[-1])
        map_const_comb = df_bal[self.x_name + ['const_comb']].drop_duplicates()
        df_bal_add = df_bal_add.join(map_const_comb.set_index(self.x_name)['const_comb'], on=self.x_name)


        df_bal = pd.concat([df_bal, df_bal_add], axis=0, sort=True)

        # if ev.select_x == m.scale_vre: join to df_bal and adjust all vre
        if self.model.vre_scale in self.x_vals.keys():
            mask_vre = df_bal.func.str.contains('vre')
            df_bal.loc[mask_vre, 'lambd'] *= df_bal.loc[mask_vre, 'vre_scale']

        varpar_neg = ['l', 'curt_p']

        df_bal.loc[df_bal.func_no_slot.isin(varpar_neg), 'lambd'] *= -1

        varpar_neg = [store.name + '_p_' + slot_name + '_lam_plot'
                      for store in self.model.storages.values()
                      for slot_name, chgdch in store.slots_map.items() if chgdch == 'chg']

        df_bal.loc[df_bal.func.isin(varpar_neg), 'lambd'] *= -1

        self.df_bal = df_bal

    def enforce_constraints(self):
        '''
        Discard solutions which violate any of the
            * positive
            * capacity
        constraints.
        TODO: Ideally this would be modular and part of the components.
        '''

        self.df_exp = self._init_constraints_active(self.df_exp)

        mask_valid = self._get_mask_valid_solutions(self.df_exp)

        self.df_exp = self.df_exp.join(mask_valid, on=mask_valid.index.names)

        self.df_exp.loc[self.df_exp.mask_valid == 0, 'lambd'] = np.nan

    def init_cost_optimum(self, df):
        ''' Adds cost optimum column to the expanded dataframe. '''

        tc = df.loc[df.func == 'tc_lam_plot'].copy()

        tc_min = (tc.groupby(self.x_name, as_index=0)
                    .apply(lambda x: x.nsmallest(1, 'lambd')))

        tc_min['is_optimum'] = True
        tc_min = tc_min.set_index(['const_comb'] + self.x_name)

        df = df[[c for c in df.columns if not c == 'is_optimum']]
        df = df.join(tc_min['is_optimum'], on=tc_min.index.names)

        df['is_optimum'] = df.is_optimum.fillna(False)

        return df.is_optimum

    def drop_non_optimal_combinations(self):
        '''
        Creates new attribute df_exp_opt with optimal constraint combs only.

        Note: This keeps all constraint combinations which are optimal
        for *some* parameter combinations.
        '''

        constrs_opt = self.df_exp.loc[self.df_exp.is_optimum]
        constrs_opt = constrs_opt['const_comb'].unique().tolist()

        mask_opt = self.df_exp.const_comb.isin(constrs_opt)
        self.df_exp_opt = self.df_exp.loc[mask_opt].copy()

    def map_func_to_slot(self):

        print('map_func_to_slot')
        func_list = self.df_exp.func.unique()

        slot_name_list = list(self.model.slots.keys())

        slot_map = {func: '+'.join([ss for ss in slot_name_list
                                    if ss in func])
                    for func in func_list}

        func_map = {func: func.replace('_None', '').replace(slot + '_lam_plot', '')
                    for func, slot in slot_map.items()}
        func_map = {func: func_new[:-1] if func_new.endswith('_') else func_new
                    for func, func_new in func_map.items()}


        slot_map = {func: slot if not slot == '' else 'global'
                    for func, slot in slot_map.items()}

        self.df_exp['slot'] = self.df_exp['func'].replace(slot_map)
        self.df_exp['func_no_slot'] = self.df_exp['func'].replace(func_map)
