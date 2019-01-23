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

import symenergy.evaluator.plotting as plotting
import grimsel.auxiliary.aux_sql_func as aql

from symenergy.auxiliary.parallelization import parallelize_df

class LambdContainer():

    def __init__(self, funcs):
        '''
        Input arguments:
            * funcs -- lists of tuples (name, function)
        '''
        for func_name, func in funcs:
            setattr(self, func_name, func)


class Evaluator(plotting.EvPlotting):
    '''
    Evaluates model results for selected
    '''

    def __init__(self, model, x_vals, drop_non_optimum=False,
                 eval_accuracy=1e-9, nthreads=None, sql_params=None):
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
        self.sql_params = sql_params

        self.drop_non_optimum = drop_non_optimum

        self.x_vals = x_vals

        self.eval_accuracy = eval_accuracy

        self.dfev = model.df_comb.copy()

        self.model.init_total_param_values()
        self.is_positive = self.model.get_all_is_positive()

        self.df_x_vals = self.get_x_vals_combs()


        if sql_params:
            self.db = sql_params['sql_db']
            self.tb = sql_params['sql_table']
            self.sc = sql_params['sql_schema']

    @property
    def x_vals(self):
        return self._x_vals

    @x_vals.setter
    def x_vals(self, x_vals):
        self._x_vals = x_vals#self.sanitize_x_vals(x_vals)
        self.x_symb = [x.symb for x in self._x_vals.keys()]
        self.x_name = [x.name for x in self.x_symb]

        self.df_x_vals = self.get_x_vals_combs()

    @property
    def df_x_vals(self):
        return self._df_x_vals

    @df_x_vals.setter
    def df_x_vals(self, df_x_vals):
        self._df_x_vals = df_x_vals.reset_index(drop=True)


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

    def get_evaluated_lambdas(self, skip_multipliers=True):
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

            print('Extracting solution for %s'%slct_eq_0, end='...')

            slct_eq = (slct_eq_0.name
                       if not isinstance(slct_eq_0, str)
                       else slct_eq_0)

            if slct_eq != 'tc':
                # function idx depends on constraint, since not all constraints
                # contain the same functions
                get_func = lambda x: self._get_func_from_idx(x, slct_eq)
                self.dfev[slct_eq] = self.dfev.apply(get_func, axis=1)

            print('substituting', end='...')
            self.dfev['%s_expr_plot'%slct_eq] = \
                        self.dfev[slct_eq].apply(self._subs_param_values)

            lambdify = lambda res_plot: sp.lambdify(self.x_symb, res_plot,
                                                    modules=['numpy'],
                                                    dummify=False)

            print('lambdify', end='...')

            self.dfev[slct_eq + '_lam_plot'] = (
                            self.dfev['%s_expr_plot'%slct_eq].apply(lambdify))
            print('done.')

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
                                  .reset_index())
        df_lam_plot = df_lam_plot.set_index(self.model.constrs_cols_neq
                                            + ['const_comb', 'func'])

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


#    def _get_expanded_row(self, lam_plot, x_vals):
#        '''
#        Apply single lambda function/row to the self.x_vals.
#
#        Input parameters:
#            * lam_plot -- Single row of the self.df_lam_plot dataframe.
#
#        Return values:
#            * Series with y values
#        '''
#
#        y_vals = [lam_plot.iloc[0](*val_row) for val_row in x_vals]
#
#        if isinstance(y_vals, float):
#            y_vals = np.ones(len(x_vals)) * y_vals
#
#        return pd.Series(y_vals, index=pd.Index(x_vals))

    def _evaluate(self, df):
        '''
        Input dataframe:
            df[['func', 'const_comb', 'lambd', 'self.x_name[0]', ...]]
        Returns expanded data for all rows in the input dataframe.
        '''


        x_dict = df.iloc[0].loc[self.x_name].to_dict()

        def process(x, report=None):
            return x.lambd_func(**x_dict)

        return df.apply(process, axis=1)

    def init_table(self, warn_existing_tables):

        sc = self.sql_params['sql_schema']
        db = self.sql_params['sql_db']
        tb = self.sql_params['sql_table']

        aql.exec_sql('CREATE SCHEMA IF NOT EXISTS %s'%(sc), db=db)

        self.sql_cols = [('func', 'VARCHAR'),
                         ('const_comb', 'VARCHAR'),
                         ('is_positive', 'SMALLINT'),
                         ('lambd', 'DOUBLE PRECISION'),
                         ('mask_valid', 'BOOLEAN'),
                         ('is_optimum', 'BOOLEAN'),
                         ] + [('"%s"'%x, 'DOUBLE PRECISION')
                              for x in self.x_name]

        aql.init_table(tb, self.sql_cols, sc, db=db,
                       warn_if_exists=warn_existing_tables)

    def evaluate_by_x(self, x, df_lam, verbose):

        df = df_lam.copy()

        t = time.time()

        if verbose:
            print(x.name, x.to_dict(), end=' -> ')
        for col in self.x_name:
            df[col] = x[col]

# ALTERNATIVE ATTEMPT: EVALUATE TC FIRST, THEN LOOP FROM LOWEST TC TO HIGHEST,
# STOP WHEN FEASIBLE; DOESN'T SEEM tO BE FASTER IN ITS CURRENT FORM
#            # evaluate total cost
#            df_tc = df.loc[df.func == 'tc_lam_plot'].copy()
#            df_tc['lambd'] = self._evaluate(df.loc[df.func == 'tc_lam_plot'])
#            # drop nan
#            df_tc = df_tc.loc[-df_tc.lambd.isnull()]
#            # sort lowest to highest
#            df_tc = df_tc.sort_values('lambd', ascending=True)
#
#            for ind, row in df_tc.iterrows():
#                print(ind)
#                df_cc = df.loc[(df.const_comb == row.const_comb)
#                             & (df.func != 'tc_lam_plot')].copy()
#
#                df_cc['lambd'] = self._evaluate(df_cc)
#
#                mask_valid = self._get_mask_valid_solutions(df_cc)
#
#                if mask_valid.mask_valid.iloc[0]:
#                    # we found the optimum
#                    df_tc_slct = df_tc.loc[df_tc.const_comb == row.const_comb]
#                    df_result = pd.concat([df_cc, df_tc_slct],
#                                          axis=0, sort=False)
#                    df_result['mask_valid'] = True
#                    df_result['is_optimum'] = True
#                    break
#                else:
#                    pass

#        else:
        df_result = df.copy()
        df_result['lambd'] = self._evaluate(df_result)

        def sanitize_unexpected_zeros(df):
            for col, func in self.model.constrs_pos_cols_vars.items():
                df.loc[(df.func == func + '_lam_plot')
                       & (df[col] != 1) & (df['lambd'] == 0),
                       'lambd'] = np.nan

        sanitize_unexpected_zeros(df_result)

        mask_valid = self._get_mask_valid_solutions(df_result)
        df_result = df_result.join(mask_valid, on=mask_valid.index.names)
        df_result['is_optimum'] = self.init_cost_optimum(df_result)

        if self.drop_non_optimum:
            df_result = df_result.loc[df_result.is_optimum]

        if self.sql_params:
            sc = self.sql_params['sql_schema']
            db = self.sql_params['sql_db']
            tb = self.sql_params['sql_table']
            cols = [col[0].replace('"', '') for col in self.sql_cols]

            aql.write_sql(df_result[cols], db, sc=sc,
                          tb=tb, if_exists='append')
            if verbose:
                print(time.time() - t)
            return None
        else:
            if verbose:
                print(time.time() - t)
            return df_result


    def _get_mask_valid_solutions(self, df):


        if __name__ == '__main__':
            df = df_result.copy()
        else:
            df = df.copy() # this is important, otherwise we change the x_vals


        mask_valid = pd.Series(True, index=df.index)

        mask_positive = self._get_mask_valid_positive(df)
        mask_valid = mask_valid & mask_positive

        mask_capacity = self._get_mask_valid_capacity(df.copy())
        mask_valid &= mask_capacity

        df['mask_valid'] = mask_valid

        # consolidate mask by constraint combination and x values
        index = self.x_name + ['const_comb']
        mask_valid = df.pivot_table(index=index,
                                    values='mask_valid',
                                    aggfunc=min)

        return mask_valid



    def call_evaluate_by_x(self, df_x, df_lam, verbose):

        eval_x = lambda x: self.evaluate_by_x(x, df_lam, verbose)

        if __name__ == '__main__':
            x = df_x.iloc[0]

        result = self.df_x_vals.apply(eval_x, axis=1)


        return result



    def after_init_table(f):
        def wrapper(self, *args, **kwargs):

            if self.sql_params and 'warn_existing_tables' in self.sql_params:
                warn_existing_tables = self.sql_params['warn_existing_tables']
            else:
                warn_existing_tables = True

            self.init_table(warn_existing_tables)

            f(self, *args, **kwargs)

        return wrapper

    @after_init_table
    def expand_to_x_vals(self, verbose=True):
        '''
        Applies evaluate_by_x to all df_x_vals rows.
            * by_x_vals -- if True: expand x_vals for all const_combs/func
                           if False: expand const_combs/func for all x_vals
        '''


        # keeping pos cols to sanitize zero equality constraints
        constrs_cols_pos = [cstr for cstr in self.model.constrs_cols_neq
                            if '_pos_' in cstr]

        # keeping cap cols to sanitize cap equality constraints
        constrs_cols_cap = [cstr for cstr in self.model.constrs_cols_neq
                            if '_cap_' in cstr]

        keep_cols = (['func', 'const_comb', 'lambd_func']
                     + constrs_cols_pos + constrs_cols_cap)
        df_lam_plot = self.df_lam_plot.reset_index()[keep_cols]

        df_lam_plot['lambd_func_hash'] = \
                df_lam_plot.lambd_func.apply(lambda x: 'func_%d'%abs(hash(x)))
        if len(df_lam_plot.lambd_func.unique()) != len(df_lam_plot):
            raise RuntimeError('lambd_func_hash has non-unique values.')


        funcs = (df_lam_plot[['lambd_func_hash', 'lambd_func']]
                            .apply(tuple, axis=1).tolist())
        self.lambd_container = LambdContainer(funcs)


        df_lam_plot = self._init_constraints_active(df_lam_plot)


        df_x = self.df_x_vals
        df_lam = df_lam_plot
        if not self.nthreads:
            df_exp_0 = self.call_evaluate_by_x(df_x, df_lam, verbose)
        else:
            func = self.call_evaluate_by_x
            nthreads = self.nthreads
            df_exp_0 = parallelize_df(df_x, func, nthreads, df_lam=df_lam)

        if not self.sql_params:
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

                slct_func = ['%s_lam_plot'%symb.name for symb in pp]

                mask_slct_func = df.func.isin(slct_func)

                # things are different depending on whether or not select_x
                # is the corresponding capacity
                if C in self.x_vals.keys():
                    val_cap = df[C.name]
                else:
                    val_cap = pd.Series(C.value, index=df.index)

                # need to add retired and additional capacity
                for addret, sign in {'add': +1, 'ret': -1}.items():
                    func_C_addret = [variab for variab in slct_func
                                     if 'C_%s_None'%addret in variab]
                    func_C_addret = func_C_addret[0] if func_C_addret else None
                    if func_C_addret:
                        mask_addret = (df.func.str.contains(func_C_addret))
                        df_C = df.loc[mask_addret].copy()
                        df_C = df_C.set_index(['const_comb'] + self.x_name)['lambd'].rename('_C_%s'%addret)
                        df = df.join(df_C, on=df_C.index.names)

                        # doesn't apply to itself, hence -mask_addret
                        val_cap.loc[-mask_addret] += \
                            + sign * df.loc[-mask_addret,
                                                     '_C_%s'%addret]

                constraint_met = pd.Series(True, index=df.index)
                constraint_met.loc[mask_slct_func] = \
                                    (df.loc[mask_slct_func].lambd
                                     * (1 - self.eval_accuracy)
                                     <= val_cap.loc[mask_slct_func])

                # delete temporary columns:
                df = df[[c for c in df.columns
                                        if not c in ['_C_ret', '_C_add']]]

                mask_valid &= constraint_met

#            self.df_exp['mv_n'] = mask_valid.copy()

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

        df_bal_add = pd.DataFrame(df_bal[self.x_name + ['const_comb']].drop_duplicates())
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

    def build_supply_table_sql(self, df=None):
        '''
        Generates a table representing the supply constraint for easy plotting.
        '''

        self.map_func_to_slot_sql()



        self.sql_cols_supply = [('func', 'VARCHAR'),
                         ('const_comb', 'VARCHAR'),
                         ('func_no_slot', 'VARCHAR'),
                         ('slot', 'VARCHAR'),
                         ('lambd', 'DOUBLE PRECISION'),
                         ] + [('"%s"'%x, 'DOUBLE PRECISION')
                              for x in self.x_name]

        self.cols_tb_supply = aql.init_table('%s_supply'%self.tb,
                                             self.sql_cols_supply, self.sc,
                                             db=self.db)

        db = self.sql_params['sql_db']

        exec_strg = '''
        INSERT INTO {sql_schema}.{sql_table}_supply ({cols})
        SELECT {cols}
        FROM {sql_schema}.{sql_table}
        WHERE is_optimum = True
            AND NOT (func LIKE 'tc%' or func LIKE 'pi_%' OR func LIKE 'lb_%');
        '''.format(**self.sql_params, cols=self.cols_tb_supply)
        aql.exec_sql(exec_strg, db=db)

        # add parameters
        par_add = ['l', 'vre']
        pars = [getattr(slot, var) for var in par_add
               for slot in self.model.slots.values() if hasattr(slot, var)]

        for par in pars:

            par_name = par.name
            slot_name = par.slot.name
            par_name_no_slot = par_name.replace('_' + slot_name, '')
            par_val = par.value

            exec_strg = '''
            WITH tb_raw AS (
              SELECT DISTINCT const_comb, {cols_x}
              FROM {sql_schema}.{sql_table}_supply
            )
            INSERT INTO {sql_schema}.{sql_table}_supply ({cols})
            SELECT
                '{par_name}'::VARCHAR AS func,
                const_comb,
                '{par_name_no_slot}'::VARCHAR AS func_no_slot,
                '{slot_name}'::VARCHAR AS slot,
                {par_val}::DOUBLE PRECISION AS lambd, {cols_x}
            FROM tb_raw
            '''.format(**self.sql_params,
                       cols=self.cols_tb_supply,
                       cols_x = ', '.join('"%s"'%x for x in self.x_name),
                       par_name=par_name,
                       par_val=par_val,
                       par_name_no_slot=par_name_no_slot,
                       slot_name=slot_name)
            aql.exec_sql(exec_strg, db=db)

        # if ev.select_x == m.scale_vre: join to df_bal and adjust all vre
        if self.model.vre_scale in self.x_vals:
            exec_strg = '''
            UPDATE {sql_schema}.{sql_table}_supply
            SET lambd = lambd * vre_scale
            WHERE func LIKE '%vre%';
            '''.format(**self.sql_params)
            aql.exec_sql(exec_strg, db=db)





        funcs_neg = []
        # add load all slots
        funcs_neg += [slot.l.name for slot in self.model.slots.values()]
        # add curtailment
        funcs_neg += [p.name for slot, p in self.model.curt.p.items()
                      ] if hasattr(self.model, 'curt') else []
        # add charging
        funcs_neg += [p.name for store in self.model.storages.values()
                      for slot, p in store.p.items()
                      if store.slots_map[slot.name] == 'chg']

        funcs_neg = ' OR func LIKE '.join("'{}%'".format(ff)
                                          for ff in funcs_neg)

        exec_strg = '''
        UPDATE {sql_schema}.{sql_table}_supply
        SET lambd = lambd * -1
        WHERE func LIKE {funcs_neg};
        '''.format(**self.sql_params, funcs_neg=funcs_neg)
        aql.exec_sql(exec_strg, db=db)


        self.df_bal = aql.read_sql(self.sql_params['sql_db'],
                                   self.sql_params['sql_schema'],
                                   self.sql_params['sql_table'])

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

    def evaluate_all(self):

        df_lam_plot = self.df_lam_plot.reset_index()[['func',
                                                      'const_comb',
                                                      'lambd_func']]

        df = pd.merge(self.df_x_vals.assign(key=1),
                      df_lam_plot.assign(key=1), on='key')

        df['lambd'] = self._evaluate(df)


    def init_cost_optimum(self, df):
        ''' Adds cost optimum column to the expanded dataframe. '''

        tc = df.loc[(df.func == 'tc_lam_plot') & df.mask_valid].copy()

        if not tc.empty:

            tc_min = (tc.groupby(self.x_name, as_index=0)
                        .apply(lambda x: x.nsmallest(1, 'lambd')))

            tc_min['is_optimum'] = True
            tc_min = tc_min.set_index(['const_comb'] + self.x_name)

            df = df[[c for c in df.columns if not c == 'is_optimum']]
            df = df.join(tc_min['is_optimum'], on=tc_min.index.names)

            df['is_optimum'] = df.is_optimum.fillna(False)

        else:

            df['is_optimum'] = False

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



    def map_func_to_slot_sql(self):

        print('map_func_to_slot')

        tb = self.sql_params['sql_table']
        sc = self.sql_params['sql_schema']
        db = self.sql_params['sql_db']

        func_list = aql.read_sql(db, sc, tb, keep=['func'], distinct=True)
        func_list = func_list['func'].tolist()

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


        all_map = ', '.join([str((func, slot_map[func], func_map[func])) for func in slot_map])

        exec_strg = '''
        ALTER TABLE {sql_schema}.{sql_table}
        ADD COLUMN IF NOT EXISTS slot VARCHAR,
        ADD COLUMN IF NOT EXISTS func_no_slot VARCHAR;
        '''.format(**self.sql_params)
        aql.exec_sql(exec_strg, db=db)

        exec_strg = '''
        UPDATE {sql_schema}.{sql_table} AS tb
        SET func_no_slot = map.func_no_slot,
            slot = map.slot
        FROM (
            WITH temp (func, slot, func_no_slot) AS (VALUES {all_map})
            SELECT * FROM temp
        ) AS map
        WHERE tb.func = map.func
        '''.format(**self.sql_params, all_map=all_map)
        aql.exec_sql(exec_strg, db=db)


    def get_readable_cc_dict(self):

        cc_h = self.model.df_comb.set_index('const_comb')[self.model.constrs_cols_neq].copy()

#        cc_h_sto = cc_h.set_index('const_comb')[[c for c in cc_h.columns if 'phs' in c and ('cap' in c or 'pos' in c)]]
        cc_h_sto = (cc_h.act_lb_phs_pos_e_None.replace({1: 'no storage', 0: ''})
                    + cc_h.act_lb_phs_p_cap_C_day.replace({1: 'max storage (day)', 0: ''})
                    + cc_h.act_lb_phs_p_cap_C_night.replace({1: 'max storage (night)', 0: ''})
                    + cc_h.act_lb_phs_e_cap_E_None.replace({1: 'max storage (e)', 0: ''}))

#        cc_h_peak = cc_h.set_index('const_comb')[[c for c in cc_h.columns if '_g_' in c and ('pos' in c)]]
        cc_h_peak = (cc_h.act_lb_g_pos_p_night.replace({0: 'peak (night)', 1: 'no peak (night)'})
                     + cc_h.act_lb_g_pos_p_day.replace({0: 'peak (day)', 1: 'no peak (day)'})).replace({'peak (night)peak (day)': 'all peak', 'no peak (night)no peak (day)': 'no peak at all'})

#        cc_h_curt = cc_h.set_index('const_comb')[[c for c in cc_h.columns if '_curt_' in c and ('pos' in c)]]
        cc_h_curt = (cc_h.act_lb_curt_pos_p_night.replace({0: 'curt (night)', 1: ''})
                     + cc_h.act_lb_curt_pos_p_day.replace({0: 'curt (day)', 1: ''})).replace({'curt (night)curt (day)': 'curtailment both'})

#        cc_h_ret = cc_h.set_index('const_comb')[[c for c in cc_h.columns if '_C_ret_' in c]]
        cc_h_ret = (cc_h.act_lb_n_C_ret_cap_C_None.replace({1: 'maximum retirement', 0: ''})
                     + cc_h.act_lb_n_pos_C_ret_None.replace({1: 'no retirement', 0: ''}))

#        cc_h_base = cc_h.set_index('const_comb')[[c for c in cc_h.columns if '_n_' in c and not 'C_ret' in c]]
        cc_h_base = (cc_h.act_lb_n_pos_p_day.replace({1: 'no base (day)', 0: ''})
                     + cc_h.act_lb_n_p_cap_C_day.replace({1: 'max base (day)', 0: ''}))

        dict_cc_h = pd.concat([cc_h_sto, cc_h_peak, cc_h_curt,
                               cc_h_ret, cc_h_base], axis=1).apply(lambda x: ' | '.join(x).replace(' |  | ', ' | '), axis=1)
        dict_cc_h = dict_cc_h.to_dict()

        return dict_cc_h

