#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the Evaluator class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import os
import sys
import gc
import py_compile
import sympy as sp
import numpy as np
from importlib import reload
from multiprocessing import current_process
import pandas as pd
import itertools
import random
from hashlib import md5
from functools import partial
import time
from sympy.utilities.lambdify import lambdastr
import symenergy

from symenergy.auxiliary.parallelization import parallelize_df
from symenergy.auxiliary.parallelization import log_time_progress
from symenergy import multiproc_params
from symenergy.auxiliary.parallelization import get_default_nworkers
from symenergy.auxiliary.parallelization import MP_COUNTER, MP_EMA
from symenergy.auxiliary import parallelization
from symenergy.auxiliary.decorators import hexdigest
from symenergy.auxiliary.io import EvaluatorCache

from symenergy.core.model import Model
from symenergy import _get_logger

logger = _get_logger(__name__)

pd.options.mode.chained_assignment = None

def log_info_mainprocess(logstr):
    if current_process().name == 'MainProcess':
        logger.info(logstr)

def _eval(func, df_x):
    '''
    Vectorized evaluation

    Parameters
    ----------
    func : pandas.Series
    df_x : pandas.DataFrame
    '''

    new_index = df_x.set_index(df_x.columns.tolist()).index
    data = func.iloc[0](*df_x.values.T)
    if not isinstance(data, np.ndarray):  # constant value --> expand
        data = np.ones(df_x.iloc[:, 0].values.shape) * data


    res = pd.DataFrame(data, index=new_index)
    MP_COUNTER.increment()

    return res


class Expander():
    '''
    Evaluates the functions in the `lambd_func` column of `df` with all
    values of the x_vals dataframe.
    '''

    def __init__(self, x_vals):

        self.df_x_vals = x_vals


    def _expand(self, df):

        logger.warning('_call_eval: Generating dataframe with length %d' % (
                        len(df) * len(self.df_x_vals)))
        if not multiproc_params['nworkers'] or multiproc_params['nworkers'] == 1:
            df_result = self._call_eval(df)
        else:
            self.nparallel = len(df)
            df_result = parallelize_df(df=df[['func', 'idx', 'lambd_func']],
                                       func=self._wrapper_call_eval)

        return df_result.rename(columns={0: 'lambd'}).reset_index()


    def _call_eval(self, df):

        df_result = (df.groupby(['func', 'idx'])
                        .lambd_func
                        .apply(_eval, df_x=self.df_x_vals))
        return df_result


    def _restore_columns(self, df_result, df):

        ind = ['func', 'idx']
        excl = ['lambd_func']
        cols = [c for c in df.columns if not c in ind + excl]
        return df_result.join(df.set_index(ind)[cols], on=ind)


    def _wrapper_call_eval(self, df):

        name, ntot = 'Vectorized evaluation', self.nparallel
        return log_time_progress(self._call_eval)(self, df, name, ntot)


    def run(self, df):

        df_result = self._expand(df)
        return self._restore_columns(df_result, df)


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class EvalAnalysis():
    '''
    Identifies optimal and infeasible solutions.
    '''

    def __init__(self, x_vals, map_col_func, dict_cap,
                 tolerance, drop_non_optimum):

        self.x_vals = x_vals
        self.map_col_func = map_col_func
        self.tolerance = tolerance
        self.drop_non_optimum = drop_non_optimum
        self.dict_cap = dict_cap

        self.x_name = list(map(lambda x: x.name, self.x_vals))


    def run(self, df):

        if not multiproc_params['nworkers'] or multiproc_params['nworkers'] == 1:
            df_exp = self._evaluate_by_x_new(df)

        else:
            group_params = self._get_optimum_group_params()
            df_split = [df for _, df in (df.groupby(group_params))]

            self.nparallel = len(df_split)
            df_exp = parallelize_df(df=df_split,
                                    func=self._wrapper_call_evaluate_by_x)
        return df_exp


    def _get_optimum_group_params(self):
        '''
        Identify groupby columns to get closest to nchunks.

        evaluate_by_x must be applied to full sets of constraint
        combinations, since constraint combinations are to be compared.
        '''

        nchunks = get_default_nworkers() * multiproc_params['chunks_per_worker']

        param_combs = \
            itertools.chain.from_iterable(itertools.combinations(self.x_vals, i)
                                          for i in range(1, len(self.x_vals) + 1))
        len_param_combs = {params: np.prod(list(len(self.x_vals[par])
                                                for par in params))
                           for params in param_combs}

        dev_param_combs = {key: abs((len_ - nchunks) / nchunks)
                           for key, len_ in len_param_combs.items()}

        group_params = min(dev_param_combs, key=lambda x: dev_param_combs[x])
        group_params = list(map(lambda x: x.name, group_params))

        return group_params


    def _get_map_sanitize(self, df):
        '''
        Identify zero values with non-binding zero constraints.
        '''

        map_ = pd.Series([True] * len(df), index=df.index)

        for col, func in self.map_col_func:
            map_new = ((df.func == func)
                       & (df[col] != 1)
                       & (df['lambd'] == 0))
            map_ &= map_new

        return map_


    def _evaluate_by_x_new(self, df):

        MP_COUNTER.increment()

        log_info_mainprocess('Sanitizing unexpected zeros.')
        df['map_sanitize'] = self._get_map_sanitize(df)
        df.loc[df.map_sanitize, 'lambd'] = np.nan

        log_info_mainprocess('Getting mask valid solutions.')
        mask_valid = self._get_mask_valid_solutions(df)
        df = df.join(mask_valid, on=mask_valid.index.names)
        df.loc[:, 'lambd'] = df.lambd.astype(float)

        log_info_mainprocess('Identify cost optimum.')
        df.loc[:, 'is_optimum'] = self.init_cost_optimum(df)

        if self.drop_non_optimum:
            df = df.loc[df.is_optimum]

        return df

#    def _call_evaluate_by_x(self, df):
#        return self._evaluate_by_x_new(df)

    def _wrapper_call_evaluate_by_x(self, df):

        name, ntot = 'Evaluate', self.nparallel
        return log_time_progress(self._evaluate_by_x_new)(self, df, name, ntot)


    def _get_mask_valid_solutions(self, df, return_full=False):
        '''
        Obtain a mask identifying valid solutions for each parameter value set
        and constraint combination.

        Indexed by x_name and constraint combination idx, *not* by function.

        Parameters
        ----------
        df : pandas.DataFrame
        return_full : bool
            if True, returns the full mask for debugging, i.e. indexed by
            functions prior to consolidation
        '''

        df = df.copy() # this is important, otherwise we change the x_vals

        mask_valid = pd.Series(True, index=df.index)
        mask_valid &= self._get_mask_valid_positive(df)
        mask_valid &= self._get_mask_valid_capacity(df.copy())

        df.loc[:, 'mask_valid'] = mask_valid

        if return_full:  # for debugging
            return df

        # consolidate mask by constraint combination and x values
        index = self.x_name + ['idx']
        mask_valid = df.pivot_table(index=index, values='mask_valid',
                                    aggfunc=min)

        return mask_valid


    def _get_mask_valid_positive(self, df):
        ''' Called by _get_mask_valid_solutions '''

        msk_pos = df.is_positive == 1
        mask_positive = pd.Series(True, index=df.index)
        mask_positive.loc[msk_pos] = df.loc[msk_pos].lambd + self.tolerance >= 0

        return mask_positive


    def _get_mask_valid_capacity(self, df):
        ''' Called by _get_mask_valid_solutions '''

        mask_valid = pd.Series(True, index=df.index)

        for C, pp in (self.dict_cap if self.dict_cap else []):

            slct_func = [symb.name for symb in pp]

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
                                 if 'C_%s_none'%addret in variab]
                func_C_addret = func_C_addret[0] if func_C_addret else None
                if func_C_addret:
                    mask_addret = (df.func.str.contains(func_C_addret))
                    df_C = df.loc[mask_addret].copy()
                    df_C = (df_C.set_index(['idx'] + self.x_name)['lambd']
                                .rename('_C_%s'%addret))
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

        return mask_valid


    def init_cost_optimum(self, df_result):
        ''' Adds binary cost optimum column to the expanded dataframe. '''

        cols = ['lambd', 'idx'] + self.x_name
        tc = df_result.loc[(df_result.func == 'tc')
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


# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Evaluator():
    '''
    Evaluates model results for selected parameter values.


    Parameters
    ----------
        model : :class:`symenergy.core.model.Model`
            SymEnergy model instance
        x_vals : dict
            dictionary ``{parameter_instance_0: iterable_of_values}``
        drop_non_optimum : bool
            if False, also keeps constraint combinations associated with
            non-optimal constraints
        tolerance : float
            absolute tolerance for constraint evaluation to allow for
            numerical inaccuracies


    Example
    -------

    .. code-block:: python

        >>> import numpy as np
        >>> from symenergy.core.model import Model
        >>> from symenergy.evaluator.evaluator import Evaluator

        >>> m = Model(curtailment=True)
        >>> m.add_slot(name='day', load=4500, vre=4500)
        >>> m.add_plant(name='n', vc0=10, vc1=1, capacity=3500)
        >>> m.add_plant(name='g', vc0=90, vc1=10)
        >>> m.generate_solve()

        >>> x_vals = {m.vre_scale: np.linspace(0, 1, 51),
                      m.comps['n'].C: [0, 1000, 3000]}
        >>> ev = Evaluator(m, x_vals=x_vals)

    The attribute ``ev.df_x_vals`` is a table with all parameter value
    combinations:

    .. code-block:: python

        >>> ev.df_x_vals
             vre_scale_none  C_n_none
        0              0.00         0
        1              0.00      1000
        2              0.00      3000
        3              0.02         0
        4              0.02      1000
        ...


    The methods

    * :func:`symenergy.evaluator.evaluator.Evaluator.get_evaluated_lambdas_parallel` and
    * :func:`symenergy.evaluator.evaluator.Evaluator.expand_to_x_vals_parallel`
    are used to perform the actual evaluation.

    .. seealso:

        :ref:`label_example_minimal`
          minimal SymEnergy example demonstrating the
          use of the evaluator class

    '''

    def __init__(self, model:Model, x_vals:dict,
                 drop_non_optimum=False, tolerance=1e-9):

        self.model = model
        self.x_vals = x_vals
        self.cache_lambd, self.cache_eval = self._get_caches()
        self.eval_analysis, self.expander = self._get_helpers(
                                                drop_non_optimum, tolerance)

        self.dfev = self._get_dfev()
        self.dict_param_values = self._get_param_values()

        # attribute name must match self.df_exp columns name
        self.is_positive = \
            self.model.constraints('expr_0', is_positivity_constraint=True)

        self.fn_temp = (os.path.basename(self.cache_lambd.fn)
                               .replace('.pickle', '_eval_temp.py'))

        self.fn_temp = os.path.abspath(os.path.join(symenergy.__file__, '..',
                                      'evaluator', self.fn_temp))


    def _get_helpers(self, drop_non_optimum, tolerance):

        map_col_func_pos = \
            self.model.constraints(('col', 'base_name'),
                                   is_positivity_constraint=True)

        dict_cap = [(cap, val) for comp in self.model.comps.values()
                    if not comp in self.model.slots.values()  # exclude slots
                    for cap, val in comp.get_constrained_variabs()]

        eval_analysis = EvalAnalysis(self.x_vals, map_col_func_pos, dict_cap,
                                     tolerance=tolerance,
                                     drop_non_optimum=drop_non_optimum)

        expander = Expander(self.df_x_vals)

        return eval_analysis, expander


    def _get_dfev(self):
        ''' Returns a modified main model DataFrame `df_comb`. Variables and
        multipliers are converted from sympy symbols to strings.'''

        cols = ['variabs_multips', 'result', 'idx', 'tc']
        dfev = self.model.df_comb[cols].copy()
        dfev.variabs_multips = dfev.variabs_multips.apply(
                                            lambda x: list(map(str, x)))

        return dfev


    @property
    def x_vals(self):
        return self._x_vals


    @x_vals.setter
    def x_vals(self, x_vals):
        x_keys_old = ([val for val in self._x_vals]
                      if hasattr(self, '_x_vals') else None)
        if x_keys_old:
            assert list(x_vals) == x_keys_old, \
                'Keys of x_vals attribute must not change.'

        frozen_params = [x.name for x in x_vals if x._is_frozen]
        assert not frozen_params, ('Encountered frozen parameters %s in '
                                   'x_vals.') % str(frozen_params)

        self._x_vals = x_vals
        self.x_symb = [x.symb for x in self._x_vals.keys()]
        self.x_name = [x.name for x in self.x_symb]
        self.x_name_str = '(%s)'%','.join(self.x_name)
        self.df_x_vals = self._get_x_vals_combs()

        self.cache_lambd, self.cache_eval = self._get_caches()

        if hasattr(self, 'eval_analysis'):
            self.eval_analysis.x_vals = self._x_vals


    def _get_caches(self):
        ''' Separate cache instances'''

        hash_lambd = self._get_evaluator_hash_name(include_x_vals=False)
        cache_lambd = EvaluatorCache(hash_lambd, 'cache_lambd')
        hash_eval = self._get_evaluator_hash_name(include_x_vals=True)
        cache_eval = EvaluatorCache(hash_eval, 'cache_eval')

        return cache_lambd, cache_eval

    @property
    def df_x_vals(self):
        return self._df_x_vals

    @df_x_vals.setter
    def df_x_vals(self, df_x_vals):
        self._df_x_vals = df_x_vals.reset_index(drop=True)
        if hasattr(self, 'expander'):
            self.expander.df_x_vals = self._df_x_vals


    def _get_list_dep_var(self, skip_multipliers=False):

        list_dep_var = ['tc']

        list_dep_var += list(map(str, self.model.variables('symb')))

        # including supply constraint even if skip_multipliers
        list_dep_var += [mlt for mlt in map(str, self.model.constraints('mlt'))
                         if (('supply' in mlt)  # only keep supply
                             if skip_multipliers
                             else True)]  # keep all

        if skip_multipliers:
            # excluding supply constraint even if
            list_dep_var = [v for v in list_dep_var
                            if (not 'lb_' in v and not 'pi_' in v)
                            or 'supply' in v]

        return list_dep_var


    def get_evaluated_lambdas(self, skip_multipliers=True):
        '''
        For each dependent variable and total cost get a lambda function
        evaluated by constant parameters. This subsequently evaluated
        for all x_pos.

        Generated attributes:
            - df_lam_func: Holds all lambda functions for each dependent
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
            expr_plot = self.dfev[slct_eq].apply(self._subs_param_values)

            lambdify = lambda res_plot: sp.lambdify(self.x_symb, res_plot,
                                                    modules=['numpy'],
                                                    dummify=False)

            logger.debug('lambdify...')

            self.dfev.loc[:, slct_eq] = expr_plot.apply(lambdify)
            logger.debug('done.')

        idx = ['idx']

        df_lam_func = self.dfev.set_index(idx).copy()[list_dep_var]

        col_names = {'level_1': 'func',
                     0: 'lambd_func'}
        df_lam_func = (df_lam_func.stack().reset_index()
                                  .rename(columns=col_names))
        df_lam_func = (df_lam_func.reset_index(drop=True)
                                  .reset_index())

        df_lam_func = df_lam_func.join(self.model.df_comb.set_index('idx')[self.model.constrs_cols_neq], on='idx')
        df_lam_func = df_lam_func.set_index(self.model.constrs_cols_neq
                                            + ['func', 'idx'])

        self.df_lam_func = df_lam_func


    def _get_func_from_idx(self, x, slct_eq):
        '''
        Get result expression corresponding to the selected variable slct_eq.

        From the result set of row x, get the expression corresponding to the
        selected variable/multiplier slct_eq. This first finds the index
        of the corresponding expression through comparison with slct_eq and
        then returns the expression itself.
        '''

        if (slct_eq in x.variabs_multips and
            not isinstance(x.result, sp.sets.EmptySet)):

            idx = x.variabs_multips.index(slct_eq)
            func = x.result[idx]

            return func

# =============================================================================
# =============================================================================

#    def _expand_dfev(self, slct_eq):
#        ''' Returns the dfev DataFrame for a single var/mlt slct_eq. '''
#
#        MP_COUNTER.increment()
#
#        df = self.dfev
#
#        get_func = partial(self._get_func_from_idx, slct_eq=slct_eq)
#        if slct_eq != 'tc':
#            df['expr'] = df.apply(get_func, axis=1)
#        else:
#            df['expr'] = df.tc
#
#        df['func'] = slct_eq
#
#        return df[['idx', 'expr', 'func']]
#
#
#    def _call_expand_dfev(self, lst_slct_eq):
#        ''' Note: here the df argument of the parallelization.parallelize_df
#        function is a list of strings, for each of which the whole self.dfev
#        is evaluated. '''
#
#        return [self._expand_dfev(slct_eq) for slct_eq in lst_slct_eq]
#
#
#    def _wrapper_call_expand_dfev(self, lst_slct_eq):
#
#        name, ntot = 'Expand by variable/multiplier', self.nparallel
#        return log_time_progress(self._call_expand_dfev)(self, lst_slct_eq,
#                                                         name, ntot)

# =============================================================================
# =============================================================================

    def _lambdify(self, expr):
        ''' Convert sympy expressions to function strings. '''

        MP_COUNTER.increment()

        return lambdastr(args=self.x_symb,
                         expr=self._subs_param_values(expr),
                         dummify=False)


    def _make_hash(self, func_str):
        ''' Generate function hash from function string. The idea is to avoid
        multiple definitions of identical functions which return e.g. constant
        zero.'''

#        salt = str(random.randint(0, 1e12))
        return '_' + md5((func_str).encode('utf-8')).hexdigest()


    def _call_lambdify(self, df):

        df['func_str'] = df.expr.apply(self._lambdify)
        df['func_hash'] = df.func_str.apply(self._make_hash)

        return df


    def _wrapper_call_lambdify(self, df):

        name, ntot = 'Lambdify expressions', self.nparallel
        return log_time_progress(self._call_lambdify)(self, df, name, ntot)

# =============================================================================
# =============================================================================

# =============================================================================
# =============================================================================

    def _replace_func_str_name(self, x):
        ''' Convert func_str to top level function strings using the names
        defined by func_hash. '''

        func_str = x.func_str
        func_hash = x.func_hash
        x_name_str = self.x_name_str

        func_str_new = ('def ' + func_hash + x_name_str
                        + ':\n    return' + func_str[len(x_name_str) + 7:])

        return func_str_new


    def _expand_results_df(self, df, skip_multipliers):
        '''
        Expands to result lists to rows.
        * zips the variable/multiplier names and the results
        * adds the total cost
        * explodes the resulting tuple

        Parameters
        ----------
        df : pandas.DataFrame
            cols ["variabs_multips", "result", "idx", "tc"]
        skip_multipliers : bool
            only include supply constraint multipliers if False, skip others
        '''

        list_dep_var = self._get_list_dep_var(skip_multipliers)

        df['result_sep'] = df.apply(lambda x: tuple((key, val) for key, val
                                     in zip(x.variabs_multips, x.result)
                                     if key in list_dep_var) + (('tc', x.tc),),
                                    axis=1)
        df = df[['idx', 'result_sep']].explode(column='result_sep')
        df[['func', 'expr']] = pd.DataFrame(df.result_sep.tolist(),
                                            index=df.index)
        df.drop('result_sep', axis=1, inplace=True)

        logger.info('Length expanded function DataFrame: %d' % len(df))

        return df

    @staticmethod
    def _write_import_function_module(fn, list_func):
        '''
        Write and import ad-hoc module containing evaluated functions.

        Parameters
        ----------
        list_func

        Returns handle of loaded module.
        '''

        try:
            os.remove(fn)
        except Exception as e :
            logger.debug(e)

        module_str = '\n'.join(list_func)
        module_str = f'from numpy import sqrt\n\n{module_str}'

        with open(fn , "w") as f:
            f.write(module_str)

        py_compile.compile(fn)

        et = __import__(os.path.basename(fn).replace('.py', ''),
                        level=1, globals={"__name__": __name__})

        return et


    def get_evaluated_lambdas_parallel(self, skip_multipliers=True,
                                        str_func=True):
        '''
        For each model variable and constraint combination, generate a function
        evaluated by all constant parameter values, but *not* by the
        varied values of the ``x_vals`` table. This results in a DataFrame
        attribute ``df_lam_func``, which is subsequently used
        (:func:`symenergy.evaluator.evaluator.Evaluator.expand_to_x_vals_parallel`)
        to generate the numerical values for variables and multipliers.


        Continuing the example from
        :class:`symenergy.evaluator.evaluator.Evaluator`:

        .. code-block:: python

            >>> import inspect
            >>> ev.cache_lambd.delete()
            >>> ev.get_evaluated_lambdas_parallel()
            >>> func = (ev.df_lam_func.set_index(['idx', 'func'])
                                      .loc[(4, 'g_p_day')]
                                      .lambd_func)
            >>> print(inspect.getsource(func))

            def _ef0ba06865119477ac40bc0b40038a25(vre_scale_none,C_n_none):
                return(-C_n_none - 4500*vre_scale_none + 4500)

        In the example above the generated example function thus depends on the
        parameters with names ``vre_scale_none`` and ``C_n_none`` (the model
        VRE scale and the power plant capacity, as specified through the
        ``x_vals`` attribute).

        **Note**: The DataFrame ``ev.df_lam_func`` is typically only used
        internally. Instead, access the tables

        * ``Model.df_comb`` to obtain the result expressions for the model
          variables
        * ``Evaluator.df_exp`` to obtain the fully evaluated numerical
          solutions

        '''

        if self.cache_lambd.file_exists:
            self.df_lam_func = self.cache_lambd.load()
            return

        dfev_exp = self._expand_results_df(self.dfev, skip_multipliers)

        self.nparallel = len(dfev_exp)
        dfev_func_str = parallelize_df(dfev_exp, self._wrapper_call_lambdify)

        # get unique function strings with function names from hashes
        list_func = (dfev_func_str[['func_hash', 'func_str']].drop_duplicates()
                                .apply(self._replace_func_str_name, axis=1))
        logger.info('Number unique function strings: %d'%len(list_func))

        # write to and read from module
        et = self._write_import_function_module(self.fn_temp, list_func)

        # retrieve eval_temp functions based on hash name
        dfev_func_str['lambd_func'] = (
            dfev_func_str.func_hash.apply(lambda x: getattr(et, x)))

        dfev_func_str = dfev_func_str.join(self.model.df_comb.set_index('idx')[
                                    self.model.constrs_cols_neq], on='idx')

        self.df_lam_func = dfev_func_str

        self.cache_lambd.write(self.df_lam_func)


    @hexdigest
    def _get_evaluator_hash_name(self, include_x_vals=False):

        hash_input = str(self.x_vals if include_x_vals else self.x_name)
        hash_input += str(self.model.get_model_hash_name())

        return hash_input



    def _init_constraints_active(self, df):
        '''
        Create binary columns depending on whether the constraints
        for each particular variable are active or not.
        '''

        def set_constr(x, lst):
            return (1 if x in map(str, getattr(self, lst)) else 0)

        lst = 'is_positive'
        for lst in ['is_positive']:

            constr_act = (df.func.apply(lambda x: set_constr(x, lst)))

            df[lst] = constr_act

        return df


    def expand_to_x_vals_parallel(self):
        '''
        Generates generates a table indexed by:

        * model variable/multiplier (column ``func``)
        * constraint combination (columns ``idx``)
        * varied parameters (columns specified by the list ``Evaluator.x_name``)

        with all numerically evaluated values of functions and multipliers.

        Other key columns are:

        * ``lambd`` numerical value
        * ``is_optimum``: boolean; if Evaluator was initialized with
          ``drop_non_optimum=True``, all non-optimal rows are dropped
        * ``mask_valid``: indicates whether the constraint combination yields
          valid results under the corresponding parameter values; see
          documentation section :ref:`label_theory_minimal` for explanations on
          infeasible constraint combinations by parameter values.

        Continuing the example from
        :func:`symenergy.evaluator.evaluator.Evaluator.get_evaluated_lambdas_parallel`:

        .. code-block:: python

            >>> ev.expand_to_x_vals_parallel()
            >>> (ev.df_exp.query('is_optimum')
                   .set_index(['func', 'idx'] + ev.x_name)[
                           ['lambd', 'is_optimum', 'mask_valid']]).head()

                                                       lambd  is_optimum  mask_valid
            func          idx vre_scale_none C_n_none
            curt_p_day    1   1.0            0           0.0        True        True
            g_p_day       1   1.0            0           0.0        True        True
            n_p_day       1   1.0            0           0.0        True        True
            pi_supply_day 1   1.0            0           0.0        True        True
            curt_p_day    2   0.0            0           0.0        True        True
            ...

        **Note:** Under some circumstances the serial evaluation is overall
        faster than the parallel approach. Serial evaluation is obtained by
        setting the SymEnergy multiprocessing *nworkers* parameter to ``None``:

        .. code-block:: python

            >>> from symenergy.auxiliary.parallelization import multiproc_params
            >>> multiproc_params('nworkers') = None
        '''

        if self.cache_eval.file_exists:
            self.df_exp = self.cache_eval.load()

        else:
            # keeping pos and cap cols to sanitize zero equality constraints
            cols_pos = self.model.constraints('col', is_positivity_constraint=True)
            cols_cap = self.model.constraints('col', is_capacity_constraint=True)
            keep_cols = (['func', 'lambd_func', 'idx'] + cols_pos + cols_cap)
            self.df_lam_func = self.df_lam_func.reset_index()[keep_cols]

            self.df_lam_func = self._init_constraints_active(self.df_lam_func)

            df_result = self.expander.run(self.df_lam_func)

            self.df_exp = self.eval_analysis.run(df_result)

            self.cache_eval.write(self.df_exp)

        self._map_func_to_slot()

        self.build_supply_table()


#    def expand_to_x_vals_loky(self):
#
#        # keeping pos and cap cols to sanitize zero equality constraints
#        cols_pos = self.model.constraints('col', is_positivity_constraint=True)
#        cols_cap = self.model.constraints('col', is_capacity_constraint=True)
#        keep_cols = (['func', 'lambd_func', 'idx'] + cols_pos + cols_cap)
#        self.df_lam_func = self.df_lam_func.reset_index()[keep_cols]
#
#        self.df_lam_func = self._init_constraints_active(self.df_lam_func)
#
#        logger.debug('_call_eval')
#        t = time.time()
#        self.nparallel = len(self.df_lam_func)
#        df_result = parallelize_df(df=self.df_lam_func[['func', 'idx', 'lambd_func']],
#                                   func=self._wrapper_call_eval)
#        df_result = df_result.rename(columns={0: 'lambd'})
#        logger.debug('done _call_eval in %fs, length df_lam %d, length df_x %d'%(time.time() - t, len(self.df_lam_func), len(self.df_x_vals)))
#
#        logger.debug('expand_to_x_vals_parallel intermediate')
#        t = time.time()
#        cols = [c for c in self.df_lam_func.columns
#                if c.startswith('act_')] + ['is_positive']
#        ind = ['func', 'idx']
#        df_result = df_result.reset_index().join(self.df_lam_func.set_index(ind)[cols],
#                                                 on=ind)
#
#        nchunks = get_default_nthreads() * parallelization.CHUNKS_PER_THREAD
#        group_params = self._get_optimum_group_params(nchunks=nchunks)
#
#        logger.debug('done expand_to_x_vals_parallel intermediate in %fs'%(time.time() - t))
#
#        logger.debug('_wrapper_call_evaluate_by_x_new')
#        t = time.time()
#        df_split = [df for _, df in (df_result.groupby(group_params))]
#        logger.debug('len(df_split): %d'%len(df_split))
#        self.nparallel = len(df_split)
#        self.df_exp = parallelize_df(df=df_split,
#                                     func=self._wrapper_call_evaluate_by_x_new)
#
#        logger.debug('done _wrapper_call_evaluate_by_x_new in %fs'%(time.time() - t))
#
#        logger.debug('_map_func_to_slot')
#        t = time.time()
#        self._map_func_to_slot()
#        logger.debug('done _map_func_to_slot in %fs'%(time.time() - t))


    def _get_x_vals_combs(self):
        '''
        Generates dataframe with all combinations of x_vals.

        Used as default by expand_to_x_vals or can be used externally to
        select subsets of
        '''

        return pd.DataFrame(list(itertools.product(*self.x_vals.values())),
                            columns=[col.name for col in self.x_vals.keys()])



    def _get_param_values(self):
        ''' Initialize dict attribute defining fixed parameter values, i.e. of
        all parameters not in `self.x_vals`. '''

        model = self.model
        x_vals = self.x_vals

        dict_param_values = model.parameters.to_dict({'symb': 'value'})

        dict_param_values = {kk: vv for kk, vv in dict_param_values.items()
                             if not kk in [x.symb for x in x_vals]}

        return dict_param_values


    def _subs_param_values(self, x):
        '''
        Substitutes all parameter values except for
        the one selected as independent variables.
        '''

        if isinstance(x, float) and np.isnan(x):
            return np.nan
        else:
            x_ret = x.subs(self.dict_param_values)
            return x_ret


#    def get_full_mask_valid(self, slct_idx):
#
#        df_slct = self.df_exp.query('idx in %s' % str(slct_idx))
#
#        return self._get_mask_valid_solutions(df=df_slct, return_full=True)






# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

    def expand_to_x_vals(self, verbose=True):
        '''
        Applies evaluate_by_x to all df_x_vals rows.
            * by_x_vals -- if True: expand x_vals for all const_combs/func
                           if False: expand const_combs/func for all x_vals
        '''

        # keeping pos cols to sanitize zero equality constraints
        cols_pos = self.model.constraints('col', is_positivity_constraint=True)

        # keeping cap cols to sanitize cap equality constraints
        cols_cap = self.model.constraints('col', is_capacity_constraint=True)

        keep_cols = (['func', 'lambd_func', 'idx'] + cols_pos + cols_cap)
        df_lam_func = self.df_lam_func.reset_index()[keep_cols]

        df_lam_func = self._init_constraints_active(df_lam_func)

        df_x = self.df_x_vals
        df_lam = df_lam_func

        logger.debug('_call_eval')
        t = time.time()

        df_result = (df_lam.groupby(['func', 'idx'])
                           .lambd_func
                           .apply(_eval, df_x=df_x))
        df_result = df_result.rename(columns={0: 'lambd'})
        logger.debug('done _call_eval in %fs, length df_lam %d, length df_x %d'%(time.time() - t, len(self.df_lam_func), len(self.df_x_vals)))


        logger.debug('expand_to_x_vals_parallel intermediate')
        t = time.time()

        cols = [c for c in df_lam.columns if c.startswith('act_')] + ['is_positive']
        ind = ['func', 'idx']
        df_result = df_result.reset_index().join(df_lam.set_index(ind)[cols],
                                                 on=ind)

        logger.debug('done expand_to_x_vals_parallel intermediate '
                     'in %fs'%(time.time() - t))

        logger.debug('_wrapper_call_evaluate_by_x_new')
        t = time.time()
        df_exp_0 = self._evaluate_by_x_new(df_result, True)
        df_exp_0 = df_exp_0.reset_index(drop=True)

        self.df_exp = df_exp_0
        self.const_comb_opt = self.df_exp.loc[self.df_exp.is_optimum, 'idx'
                                             ].unique().tolist()

        logger.debug('done _wrapper_call_evaluate_by_x_new in %fs'%(time.time() - t))


        logger.debug('_map_func_to_slot')
        t = time.time()
        self._map_func_to_slot()
        logger.debug('done _map_func_to_slot in %fs'%(time.time() - t))

        self.build_supply_table()


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
        varpar_neg = [store.name + '_p' + chgdch + '_' + slot_name
                      for store in self.model.storages.values()
                      for chgdch, slots_names in store.slots_map.items()
                      for slot_name in slots_names if chgdch == 'chg']

        df_bal.loc[df_bal.func.isin(varpar_neg), 'lambd'] *= -1

        self.df_bal = df_bal



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

        logger.info('Mapping model variables to time slots')
        func_list = self.df_exp.func.unique()

        slot_name_list = list(self.model.slots.keys())

        slot_map = {func: '+'.join([ss for ss in slot_name_list
                                    if ss in func])
                    for func in func_list}

        func_map = {func: func.replace('_None', '').replace(slot, '')
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

