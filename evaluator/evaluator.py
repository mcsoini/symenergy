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

from symenergy.core.model import Model
from symenergy import _get_logger

logger = _get_logger(__name__)



class LambdContainer():

    def __init__(self, funcs):
        '''
        Input arguments:
            * funcs -- lists of tuples (name, function)
        '''
        for func_name, func in funcs:
            setattr(self, func_name, func)


class Evaluator():
    '''
    Evaluates model results for selected
    '''

    def __init__(self, model:Model, x_vals:dict, drop_non_optimum=False,
                 tolerance=1e-9):
        '''
        Parameters
        ----------
            model : :class:`symenergy.core.model.Model`
            x_vals : dict or list
                either dictionary `{parameter_name: list_of_value}`
            tolerance : float
                absolute tolerance for constraint evaluation
        '''

        self.model = model

        self.drop_non_optimum = drop_non_optimum

        self._x_vals = None
        self.x_vals = x_vals

        self.tolerance = tolerance

        self.dfev = self.model.df_comb.copy()

        self.dict_param_values = self.model.parameters.to_dict({'symb': 'value'})

        self.param_values = self.model

        # attribute name must match self.df_exp columns name
        self.is_positive = self.model.constraints('expr_0',
                                               is_positivity_constraint=True)

#        self._get_evaluated_lambdas()

    @property
    def x_vals(self):
        return self._x_vals

    @x_vals.setter
    def x_vals(self, x_vals):
        x_keys_old = [val for val in self._x_vals] if self._x_vals else None
        x_keys = list(x_vals)
        if x_keys_old:
            assert x_keys == x_keys_old, \
                'Keys of x_vals attribute must not change.'

        frozen_params = [x.name for x in x_vals if x._is_frozen]
        assert not frozen_params, ('Encountered frozen parameters %s in '
                                   'x_vals.') % str(frozen_params)

        self._x_vals = x_vals#self.sanitize_x_vals(x_vals)
        self.x_symb = [x.symb for x in self._x_vals.keys()]
        self.x_name = [x.name for x in self.x_symb]

        self.df_x_vals = self._get_x_vals_combs()

    @property
    def df_x_vals(self):
        return self._df_x_vals

    @df_x_vals.setter
    def df_x_vals(self, df_x_vals):
        self._df_x_vals = df_x_vals.reset_index(drop=True)


    def _get_list_dep_var(self, skip_multipliers=False):

        list_dep_var = list(map(str, self.model.constraints('mlt')
                                     + self.model.variables('symb')))

        list_dep_var += ['tc']

        if skip_multipliers:
            list_dep_var = [v for v in list_dep_var
                            if (not 'lb_' in v and not 'pi_' in v)
                            or 'supply' in v]

        return list_dep_var


    def _get_evaluated_lambdas(self, skip_multipliers=True):
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

        slct_eq = ('n_p_day' if 'n_p_day' in list_dep_var
                     else list_dep_var[0])
        for slct_eq in list_dep_var:

            logger.info('Generating lambda functions for %s.'%slct_eq)

            if slct_eq != 'tc':
                # function idx depends on constraint, since not all constraints
                # contain the same functions
                get_func = lambda x: self._get_func_from_idx(x, slct_eq)
                self.dfev.loc[:, slct_eq] = self.dfev.apply(get_func, axis=1)

            logger.debug('substituting...')
            self.dfev.loc[:, '%s_expr_plot'%slct_eq] = \
                        self.dfev[slct_eq].apply(self._subs_param_values)

            lambdify = lambda res_plot: sp.lambdify(self.x_symb, res_plot,
                                                    modules=['numpy'],
                                                    dummify=False)

            logger.debug('lambdify...')

            self.dfev.loc[:, slct_eq + '_lam_plot'] = (
                            self.dfev['%s_expr_plot'%slct_eq].apply(lambdify))
            logger.debug('done.')

        idx = self.model.constrs_cols_neq + ['idx']
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
                                            + ['func', 'idx'])

        self.df_lam_plot = df_lam_plot



    def _get_x_vals_combs(self):
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

            func = x.result[idx]

            return func

    def _subs_param_values(self, x):
        '''
        Substitutes all parameter values except for
        the one selected as independent variables.
        '''

        if isinstance(x, float) and np.isnan(x):
            return np.nan
        else:
            x_ret = x.subs({kk: vv for kk, vv
                            in self.dict_param_values.items()
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

        df.loc[:, 'mask_valid'] = mask_valid

        # consolidate mask by constraint combination and x values
        index = self.x_name + ['idx']
        mask_valid = df.pivot_table(index=index,
                                    values='mask_valid',
                                    aggfunc=min)

        return mask_valid


    def call_evaluate_by_x_new(self, df_x, df_lam, verbose):

        t = time.time()

        new_index = df_x.set_index(df_x.columns.tolist()).index

        def _eval(func):
            data = func.iloc[0](*df_x.values.T)
            if not isinstance(data, np.ndarray):  # constant value --> expand
                data = np.ones(df_x.iloc[:, 0].values.shape) * data
            return pd.DataFrame(data, index=new_index)

        df_result = df_lam.groupby(['func', 'idx']).lambd_func.apply(_eval)
        df_result = df_result.rename(columns={0: 'lambd'})

        cols = [c for c in df_lam.columns if c.startswith('act_')] + ['is_positive']
        ind = ['func', 'idx']
        df_result = df_result.reset_index().join(df_lam.set_index(ind)[cols],
                                                 on=ind)

        def sanitize_unexpected_zeros(df):
            map_col_func = self.model.constraints(('col', 'base_name'),
                                              is_positivity_constraint=True)
            for col, func in map_col_func:
                df.loc[(df.func == func + '_lam_plot')
                       & (df[col] != 1)
                       & (df['lambd'] == 0), 'lambd'] = np.nan

        sanitize_unexpected_zeros(df_result)

        mask_valid = self._get_mask_valid_solutions(df_result)
        df_result = df_result.join(mask_valid, on=mask_valid.index.names)
        df_result.loc[:, 'lambd'] = df_result.lambd.astype(float)
        df_result.loc[:, 'is_optimum'] = self.init_cost_optimum(df_result)

        if self.drop_non_optimum:
            df_result = df_result.loc[df_result.is_optimum]

        if verbose:
            logger.info(time.time() - t)
        return df_result


    def expand_to_x_vals(self, verbose=True):
        '''
        Applies evaluate_by_x to all df_x_vals rows.
            * by_x_vals -- if True: expand x_vals for all const_combs/func
                           if False: expand const_combs/func for all x_vals
        '''

        # keeping pos cols to sanitize zero equality constraints
        constrs_cols_pos = self.model.constraints('col',
                                                  is_positivity_constraint=True)

        # keeping cap cols to sanitize cap equality constraints
        constrs_cols_cap = self.model.constraints('col',
                                                  is_capacity_constraint=True)

        keep_cols = (['func', 'lambd_func', 'idx']
                     + constrs_cols_pos + constrs_cols_cap)
        df_lam_plot = self.df_lam_plot.reset_index()[keep_cols]


        df_lam_plot = self._init_constraints_active(df_lam_plot)

        df_x = self.df_x_vals
        df_lam = df_lam_plot
        df_exp_0 = self.call_evaluate_by_x_new(df_x, df_lam, verbose)
        df_exp_0 = df_exp_0.reset_index(drop=True)

        self.df_exp = df_exp_0
        self.const_comb_opt = self.df_exp.loc[self.df_exp.is_optimum, 'idx'
                                             ].unique().tolist()

        self._map_func_to_slot()

    def _init_constraints_active(self, df):
        '''
        Create binary columns depending on whether the plant constraints
        are active or not.
        '''

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
        mask_positive.loc[msk_pos] = df.loc[msk_pos].lambd + self.tolerance >= 0

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
                        df_C = df_C.set_index(['idx'] + self.x_name)['lambd'].rename('_C_%s'%addret)
                        df = df.join(df_C, on=df_C.index.names)

                        # doesn't apply to itself, hence -mask_addret
                        val_cap.loc[-mask_addret] += \
                            + sign * df.loc[-mask_addret,
                                                     '_C_%s'%addret]

                constraint_met = pd.Series(True, index=df.index)
                constraint_met.loc[mask_slct_func] = \
                                    (df.loc[mask_slct_func].lambd
                                     * (1 - self.tolerance)
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
        drop = ['tc_', 'pi_', 'lb_']
        df_bal = df_bal.loc[-df_bal.func.str.contains('|'.join(drop))]

        df_bal = df_bal[['func', 'idx', 'func_no_slot',
                         'slot', 'lambd'] + self.x_name]

        # map to pwr/erg
        list_erg_var = [var_e.name for store in self.model.storages.values()
                        for var_e in store.e.values()]
        list_erg_func = [f for f in df_bal.func.unique()
                         if any(f.startswith(var_e)
                                for var_e in list_erg_var)]
        df_bal.loc[:, 'pwrerg'] = (df_bal.assign(pwrerg='erg').pwrerg
                                      .where(df_bal.func.isin(list_erg_func),
                                         'pwr'))

        # add parameters
        par_add = ['l', 'vre']
        pars = [getattr(slot, var) for var in par_add
               for slot in self.model.slots.values() if hasattr(slot, var)]
        pars_x = [p for p in pars if p.name in self.x_name]
        pars = [p for p in pars if not p.name in self.x_name]

        df_bal_add = pd.DataFrame(df_bal[self.x_name + ['idx']]
                                    .drop_duplicates())
        for par in pars:
            df_bal_add.loc[:, par.name] = par.value

        for par in pars_x:
            df_bal_add.loc[:, 'y_' + par.name] = df_bal_add[par.name]

        df_bal_add = df_bal_add.set_index(self.x_name + ['idx']).stack().rename('lambd').reset_index()
        df_bal_add = df_bal_add.rename(columns={'level_%d'%(1 + len(self.x_name)): 'func'})
        df_bal_add.func = df_bal_add.func.apply(lambda x: x.replace('y_', ''))
        df_bal_add.loc[:, 'func_no_slot'] = df_bal_add.func.apply(lambda x: '_'.join(x.split('_')[:-1]))
        df_bal_add.loc[:, 'slot'] = df_bal_add.func.apply(lambda x: x.split('_')[-1])
        df_bal_add.loc[:, 'pwrerg'] = 'pwr'

        df_bal = pd.concat([df_bal, df_bal_add], axis=0, sort=True)

        # if ev.select_x == m.scale_vre: join to df_bal and adjust all vre
        if self.model.vre_scale in self.x_vals:
            mask_vre = df_bal.func.str.contains('vre')
            df_bal.loc[mask_vre, 'lambd'] *= df_bal.loc[mask_vre, 'vre_scale_none']

        # negative by func_no_slot
        varpar_neg = ['l', 'curt_p']
        df_bal.loc[df_bal.func_no_slot.isin(varpar_neg), 'lambd'] *= -1

        # negative by func
        varpar_neg = [store.name + '_p' + chgdch + '_' + slot_name + '_lam_plot'
                      for store in self.model.storages.values()
                      for chgdch, slots_names in store.slots_map.items()
                      for slot_name in slots_names if chgdch == 'chg']

        df_bal.loc[df_bal.func.isin(varpar_neg), 'lambd'] *= -1

        self.df_bal = df_bal


    def init_cost_optimum(self, df_result):
        ''' Adds cost optimum column to the expanded dataframe. '''

        cols = ['lambd', 'idx'] + self.x_name
        tc = df_result.loc[(df_result.func == 'tc_lam_plot')
                           & df_result.mask_valid, cols].copy()

        if not tc.empty:

            tc_min = (tc.groupby(self.x_name, as_index=0)
                        .apply(lambda x: x.nsmallest(1, 'lambd')))

            def get_cost_optimum_single(df):
                df = df.sort_values('lambd')
                df.loc[:, 'is_optimum'] = False
                df.iloc[0, -1] = True
                return df[['is_optimum']]

            mask_is_opt = (tc.set_index('idx')
                             .groupby(self.x_name)
                             .apply(get_cost_optimum_single))

            df_result = df_result.join(mask_is_opt, on=mask_is_opt.index.names)

            # mask_valid == False have is_optimum == NaN at this point
            df_result.is_optimum.fillna(False, inplace=True)

        else:

            df_result.loc[:, 'is_optimum'] = False

        return df_result.is_optimum

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


    def _map_func_to_slot(self):

        logger.debug('map_func_to_slot')
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

        self.df_exp.loc[:, 'slot'] = self.df_exp['func'].replace(slot_map)
        self.df_exp.loc[:, 'func_no_slot'] = self.df_exp['func'].replace(func_map)


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

