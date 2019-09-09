#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the main Model class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import os
import sys

import itertools
import pandas as pd
import sympy as sp
import numpy as np
import hashlib
import time
from sympy.tensor.array import derive_by_array


import symenergy
from symenergy.assets.plant import Plant
from symenergy.assets.storage import Storage
from symenergy.assets.curtailment import Curtailment
from symenergy.core.constraint import Constraint
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter
from symenergy.auxiliary.parallelization import parallelize_df
from symenergy import _get_logger
from symenergy.patches.sympy_linsolve import linsolve



logger = _get_logger(__name__)

logger.warning('!!! Monkey-patching sympy.linsolve !!!')
sp.linsolve = linsolve


if __name__ == '__main__':
    sys.exit()


import multiprocessing

class Counter():
    def __init__(self):
        self.val = multiprocessing.Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


class Model:

    def __init__(self, nthreads=None, curtailment=False):

        self.plants = {}
        self.slots = {}
        self.storages = {}

        self.comps = []

        self.nthreads = nthreads

        # global vre scaling parameter, set to 1; used for evaluation
        self.vre_scale = Parameter('vre_scale', noneslot, 1)

        self.noneslot = noneslot

        self.curtailment = curtailment

        self.ncomb = None  # determined through construction of self.df_comb
        self.nress = None  # number of valid results

        self.ema_solve = 0  # exponentially moving average of solving time

    def update_component_list(f):
        def wrapper(self, *args, **kwargs):
            f(self, *args, **kwargs)

            self.comps = self.plants.copy()
            self.comps.update(self.slots)
            self.comps.update(self.storages)
            self.init_curtailment()

            self.collect_component_constraints()
            self.init_supply_constraints()
            self.init_total_param_values()
            self.get_variabs_params()
            self.init_total_cost()

            self.init_supply_constraints()

            self.init_cache_pickle_filename()

        return wrapper

    def init_curtailment(self):

        if self.curtailment:
            self.curt = Curtailment('curt', self.slots)
            self.comps['curt'] = self.curt

    def init_cache_pickle_filename(self):

        fn = '%s.pickle'%self.get_name()
        fn = os.path.join(list(symenergy.__path__)[0], 'cache', fn)

        self.cache_fn = fn


    @property
    def df_comb(self):
        return self._df_comb

    @df_comb.setter
    def df_comb(self, df_comb):
        self._df_comb = df_comb.reset_index(drop=True)

    @update_component_list
    def add_storage(self, name, *args, **kwargs):
        ''''''

        self.storages.update({name: Storage(name, **kwargs)})

    @update_component_list
    def add_plant(self, name, *args, **kwargs):

        self.plants.update({name: Plant(name, **kwargs)})

    @update_component_list
    def add_slot(self, name, *args, **kwargs):

        assert isinstance(name, str), 'Slot name must be string.'

        self.slots.update({name: Slot(name, **kwargs)})

    def init_total_param_values(self):
        '''
        Generates dictionary {parameter object: parameter value}
        for all parameters for all components.
        '''

        self.param_values = {symb: val
                             for comp in self.comps.values()
                             for symb, val
                             in comp.get_params_dict(('symb',
                                                     'value')).items()}

        self.param_values.update({self.vre_scale.symb: self.vre_scale.value})

    def init_total_cost(self):

        self.tc = sum(p.cc for p in self.plants.values())
        self.lagrange_0 = (self.tc
                           + sum([cstr.expr for cstr in self.cstr_supply]))
        self.lagrange_0 += sum([cstr.expr for cstr in self.constrs
                                if cstr.is_equality_constraint])


    def init_supply_constraints(self):
        '''
        Defines a dictionary cstr_load {slot: supply constraint}
        '''

        self.cstr_supply = {}

        for slot in self.slots.values():

            cstr_supply = Constraint('load_%s'%(slot.name),
                                     multiplier_name='pi', slot=slot,
                                     is_equality_constraint=True)
            cstr_supply.expr = \
                    self.get_supply_constraint_expr(cstr_supply)

            self.cstr_supply[cstr_supply] = slot



    def get_supply_constraint_expr(self, cstr):
        '''
        Initialize the load constraints for a given time slot.
        Note: this accesses all plants, therefore method of the model class.
        '''

        slot = cstr.slot

        total_chg = sum(store.pchg[slot]
                        for store in self.storages.values()
                        if slot in store.pchg)
        total_dch = sum(store.pdch[slot]
                        for store in self.storages.values()
                        if slot in store.pdch)

        equ = \
        cstr.mlt * (slot.l.symb
                    - slot.vre.symb * self.vre_scale.symb
                    + total_chg
                    - total_dch
                    - sum(plant.p[slot] for plant
                          in self.plants.values()))

        if self.curtailment:
            equ += cstr.mlt * self.curt.p[slot]

        return equ

    def generate_solve(self):

        if os.path.isfile(self.cache_fn):
            log_str1 = 'Loading from pickle file %s.'%self.cache_fn
            log_str2 = 'Please delete this file to re-solve model.'
            logger.info('*'*max(len(log_str1), len(log_str2)))
            logger.info('*'*max(len(log_str1), len(log_str2)))
            logger.info(log_str1)
            logger.info(log_str2)
            logger.info('*'*max(len(log_str1), len(log_str2)))
            logger.info('*'*max(len(log_str1), len(log_str2)))
            self.df_comb = pd.read_pickle(self.cache_fn)

        else:

            self.init_constraint_combinations()

#            self.remove_mutually_exclusive_combinations()

#            self.delete_cached()

            self.define_problems()

            self.solve_all()

            self.filter_invalid_solutions()

            self.generate_total_costs()

            self.fix_stored_energy()
            self.df_comb.to_pickle(self.cache_fn)


    def generate_total_costs(self):
        '''
        Substitute result variable expressions into total costs


        '''
        df = list(zip(self.df_comb.result,
                      self.df_comb.variabs_multips,
                      self.df_comb.idx))
        if not self.nthreads:
            self.df_comb['tc'] = self.call_subs_tc(df)
        else:
            func = self.call_subs_tc
            nthreads = self.nthreads
            self.df_comb['tc'] = parallelize_df(df, func, nthreads)



    def get_result_dict(self, row, string_keys=False):
        '''
        Combines the variabs_multips and the result iterables into a dict.

        Keyword argument:
            * x -- df_comb table row, must contain columns
                   'variabs_multibs', 'result'
            * string_keys -- boolean; if True, return symbol names as keys
                             rather than symbols
        '''

        dict_res = {str(var) if string_keys else var: res
                    for var, res
                    in zip(row.variabs_multips, row.result[0])}
        return dict_res


    def fix_stored_energy(self):

        if __name__ == '__main__':
            x = self.df_comb.iloc[0]
            row = x

        if self.storages:
            logger.info('Recalculating stored energy.')
            self.df_comb['result'] = self.df_comb.apply(
                                        self._fix_stored_energy, axis=1)
        else:
            logger.warning('Skipping stored energy recalculation. '
                           'Model does not contain storage assets.')


    def _fix_stored_energy(self, x):
        '''
        Recalculates stored energy from power results.

        Input parameters:
            * x -- df_comb table row, must contain columns
                   'variabs_multibs', 'result'
            * name -- storage object name

        TODO: Should be done by storage class.
        '''

        dict_var = self.get_result_dict(x, True)

        for store in self.storages.values():
            sum_chg = sum(dict_var['%s_p_%s'%(store.name, chg_slot)]
                          * (self.slots[chg_slot].weight
                             if self.slots[chg_slot].weight else 1)
                          for chg_slot in
                          tuple(slot for slot, cd in store.slots_map.items()
                                if cd == 'chg'))

            dict_var['%s_e_None'%store.name] = sum_chg * store.eff.symb**0.5

        return [[dict_var[str(var)] for var in x.variabs_multips]]


    def collect_component_constraints(self):
        '''
        Note: Doesn't include supply constraints, which are a model attribute.
        '''

        self.constrs = {}
        for comp in self.comps.values():
            for cstr in comp.get_constraints(by_slot=True, names=False):
                self.constrs[cstr] = comp

        # dictionary {column name: constraint object}
        self.constrs_dict = {cstr.col: cstr for cstr in self.constrs.keys()}

        # list of constraint columns
        self.constrs_cols = [cstr.col for cstr in self.constrs.keys()]

        self.constrs_neq = [cstr for cstr in self.constrs
                            if not cstr.is_equality_constraint]
        self.constrs_cols_neq = [cstr.col for cstr
                                 in self.constrs_neq]

        self.constrs_pos_cols_vars = {cstr.col: cstr.var_name
                                      for cstr in self.constrs
                                      if '_pos_' in cstr.col}

    def init_constraint_combinations(self):
        '''
        Gathers all non-equal component constraints,
        calculates cross-product of all
        (binding, non-binding) combinations and instantiates dataframe.
        '''

        list_dfcomb = []
        for comp in self.comps.values():
            list_dfcomb.append(comp.get_constraint_combinations())

        list_dfcomb = [df for df in list_dfcomb if not df.empty]

        dfcomb = pd.DataFrame({'dummy': 1}, index=[0])

        for df in list_dfcomb:
            dfcomb = pd.merge(dfcomb, df.assign(dummy=1),
                              on='dummy', how='outer')

        self.df_comb = dfcomb.drop('dummy', axis=1)


    def get_variabs_params(self):
        '''
        Generate lists of parameters and variables.

        Gathers all parameters and variables from its components.
        This is needed for the definition of the linear equation system.
        '''

        self.params = {par: comp
                       for comp in self.comps.values()
                       for par in comp.get_params_dict()}

        # add time-dependent variables
        self.variabs = {var: comp
                        for comp in self.comps.values()
                        for var in comp.get_variabs()}

        # parameter multips
        self.multips = {cstr.mlt: comp for cstr, comp in self.constrs.items()}
        # supply multips
        self.multips.update({cstr.mlt: slot
                             for cstr, slot in self.cstr_supply.items()})

    def get_variabs_multips_slct(self, lagrange):
        '''
        Returns all relevant variables and multipliers for this model.

        Starting from the complete set of variables and multipliers, they are
        filtered depending on whether they occur in a specific lagrange
        function.

        Parameters:
            * lagrange -- sympy expression; lagrange function

        Return values:
            * variabs_slct --
            * variabs_time_slct --
            * multips_slct --
        '''

        lfs = lagrange.free_symbols
        return [ss for ss in lfs if ss in self.variabs or ss in self.multips]



#    def print_row(self, index, matrix_dim):
#
#        matrix_dim = 'Matrix %s'%'x'.join(map(str, matrix_dim))
#
#        strg = 'Constr. comb. %d out of %d: %s.'%(index, self.n_comb,
#                                                     matrix_dim)
#
#        logger.info(strg)


    @wrapt.decorator
    def update_ema_time(f, self, args, kwargs):

        idx = args[-1]

        if not idx % 1:
            t = time.time()
            result = f(*args, **kwargs)
            t = time.time() - t
            self.ema_solve = 0.7 * self.ema_solve + 0.3 * t
        else:
            result = f(*args, **kwargs)

#        self.n_solved.increment()

        if not idx % 1:
            logger.info('Average solution time: {:.4f}, n=/{}'.format(self.ema_solve,
#                                                                        self.n_solved,
                                                                        self.n_comb))

        return result

    @update_ema_time
    def solve(self, lagrange, variabs_multips_slct, index):

        mat = derive_by_array(lagrange, variabs_multips_slct)
        mat = sp.Matrix(mat).expand()
        A, b = sp.linear_eq_to_matrix(mat, variabs_multips_slct)
        solution = sp.linsolve((A, b), variabs_multips_slct)

        return None if isinstance(solution, sp.sets.EmptySet) else solution


    def call_solve_df(self, df):
        ''' Applies to dataframe. '''

        logger.info('Calling call_solve_df on list with length %d'%len(df))
        return [self.solve(lag, var, idx) for lag, var, idx in df]


    def solve_all(self):

        df = list(zip(self.df_comb.lagrange,
                      self.df_comb.variabs_multips,
                      self.df_comb.idx))

        self.ema_solve = 0
#        self.n_solved = Counter()


        if __name__ == '__main__':
            lagrange, variabs_multips_slct, index = df[0]

        if not self.nthreads:
            self.df_comb['result'] = self.call_solve_df(df)
        else:
            func = self.call_solve_df
            nthreads = self.nthreads
            self.df_comb['result'] = parallelize_df(df, func, nthreads)

    def subs_total_cost(self, result, var_mlt_slct, idx):
        '''
        Substitutes solution into TC variables.
        This expresses the total cost as a function of the parameters.
        '''

        if not idx % 10:
            logger.info('Substituting parameters into total '
                        'cost %d/%d'%(idx, self.nress))

        dict_var = {var: list(result)[0][ivar]
                    if not isinstance(result, sp.sets.EmptySet)
                    else np.nan for ivar, var
                    in enumerate(var_mlt_slct)}

        return self.tc.copy().subs(dict_var)


    def call_subs_tc(self, df):

        logger.info('Calling call_subs_tc on list with length %d'%len(df))
        return [self.subs_total_cost(res, var, idx) for res, var, idx in df]


    def combine_constraint_names(self, df):

        constr_name = pd.DataFrame(index=df.index)
        for const in self.constrs_cols_neq:

            constr_name[const] = const + '=' + df[const].astype(str)

        join = lambda x: ', '.join(x)
        df['const_comb'] = constr_name.apply(join, axis=1)

        return df

    def construct_lagrange(self, row):

        if not row.name % 1000:
            print(row.name)

        lagrange = self.lagrange_0

        active_cstrs = row[row == 1].index.values
        lagrange += sum(self.constrs_dict[cstr_name].expr
                        for cstr_name in active_cstrs)

        return lagrange



    def call_construct_lagrange(self, df):
        '''
        Top-level method for parallelization of construct_lagrange.
        '''
        logger.info('Calling call_construct_lagrange on DataFrame '
              'with length %d'%len(df))
        return df.apply(self.construct_lagrange, axis=1).tolist()



    def call_get_variabs_multips_slct(self, df):

        logger.info('Calling get_variabs_multips_slct on DataFrame '
              'with length %d'%len(df))

        return list(map(self.get_variabs_multips_slct, df))

    def define_problems(self):
        '''
        For each combination of constraints, get the lagrangian
        and the variables.
        '''

        logger.info('Defining lagrangians...')
        if not self.nthreads:
            df = self.df_comb[self.constrs_cols_neq]
            self.df_comb['lagrange'] = self.call_construct_lagrange(df)
        else:
            df = self.df_comb[self.constrs_cols_neq]
            func = self.call_construct_lagrange
            nthreads = self.nthreads
            self.df_comb['lagrange'] = parallelize_df(df, func, nthreads)



        logger.info('Getting selected variables/multipliers...')
        df = self.df_comb.lagrange
        self.list_variabs_multips = self.call_get_variabs_multips_slct(df)
        self.df_comb['variabs_multips'] = self.list_variabs_multips

        # get index for reporting
        self.df_comb = self.df_comb[[c for c in self.df_comb.columns
                                     if not c == 'idx']].reset_index()
        self.df_comb = self.df_comb.rename(columns={'index': 'idx'})

        # get length for reporting
        self.n_comb = self.df_comb.iloc[:, 0].size


    def get_mask_empty_solution(self):
        '''
        Infeasible solutions are empty.
        '''

        mask_empty = self.df_comb.result.isnull()

        return mask_empty

    def get_mask_linear_dependencies(self):
        '''
        Solutions of problems containing linear dependencies.

        In case of linear dependencies SymPy returns solutions containing
        variables which we are actually solving for. To fix this, we
        differentiate between two cases:
            1. All corresponding solutions belong to the same component.
               Overspecification occurs if the variables of the same component
               depend on each other but are all zero. E.g. charging,
               discharging, and stored energy in the case of storage.
               They are set to zero.
            2. Linear dependent variables belonging to different components.
               This occurs if the model is underspecified, e.g. if it doesn't
               matter which component power is used. Then the solution can
               be discarded without loss of generality. All cases will still
               be captured by other constraint combinations.

        Returns:
            * Series mask with values: 0 -> no variables in solutions; 1 ->
                single variable in solution: set to zero; > 1 -> mixed
                interdependency: drop solution; NaN: empty solution set

        '''

        check_vars_left = lambda x: any(var in x.result.free_symbols
                                        for var in x.variabs_multips)
        mask_lindep = self.df_comb.apply(check_vars_left, axis=1)

        # get residual variables
        if __name__ == '__main__':
            x = self.df_comb.iloc[0]

        # mask with non-empty solutions
        not_empty = lambda x: not isinstance(x, sp.sets.EmptySet)
        mask_valid = self.df_comb.result.apply(not_empty)

        # for each individual solution, get residual variables/multipliers
        get_residual_vars = \
            lambda x: tuple([var for var in res.free_symbols
                                 if var in x.variabs_multips]
                            for nres, res in enumerate(list(x.result)[0]))

        res_vars = self.df_comb[['result', 'variabs_multips']].copy()
        res_vars.loc[mask_valid, 'res_vars'] = \
                self.df_comb.loc[mask_valid].apply(get_residual_vars, axis=1)

        if __name__ == '__main__':
            x = res_vars.iloc[1]

        # add solution variable itself to all non-empty lists
        add_solution_var = \
            lambda x: tuple(x.res_vars[nres_vars]
                                + [x.variabs_multips[nres_vars]]
                            if res_vars else []
                            for nres_vars, res_vars in enumerate(x.res_vars))

        res_vars.loc[mask_valid, 'res_vars'] = \
                res_vars.loc[mask_valid].apply(add_solution_var, axis=1)

        # get component corresponding to variable
        comp_dict_varmtp = self.multips.copy()
        comp_dict_varmtp.update(self.variabs)

        get_comps = lambda x: tuple((list(set([comp_dict_varmtp[var]
                                              for var in res_vars])))
                                    for res_vars in x.res_vars
                                    if res_vars)
        res_vars.loc[mask_valid, 'res_comps'] = \
                res_vars.loc[mask_valid].apply(get_comps, axis=1)

        # maximum unique component number
        def get_max_nunq(x):
            nmax = 0
            if x.res_comps:
                nmax = max(len(set(comp_list)) for comp_list in x.res_comps)
            return nmax

        res_vars.loc[mask_valid, 'mask_res_unq'] = res_vars.loc[mask_valid].apply(get_max_nunq, axis=1)

        return res_vars.mask_res_unq



    def get_name(self):
        '''
        Returns a unique hashed model name based on the constraint names.
        '''

        list_slots = ['%s_%s'%(slot.name, str(slot.weight)) for slot in self.slots.values()]
        list_slots.sort()
        list_cstrs = [cstr.base_name for cstr in self.constrs]
        list_cstrs.sort()
        list_param = [par.name for par in self.params]
        list_param.sort()
        list_cstrs = [par.name for par in self.variabs]
        list_cstrs.sort()
        list_multips = [par.name for par in self.multips]
        list_multips.sort()

        m_name = '_'.join(list_cstrs + list_param + list_cstrs + list_multips
                          + list_slots)

        m_name = hashlib.md5(m_name.encode('utf-8')).hexdigest()[:12].upper()

        return m_name

    def cache(self, list_infeas):
        '''
        Saves list of infeasible constraint combinations to file on disk.
        '''

        if not os.path.isfile(self.cache_fn):
            list_infeas.to_csv(self.cache_fn, index=False)






#    def delete_cached(self):
#
#        try:
#            list_infeas = pd.read_csv(self.cache_fn, header=None)
#            list_infeas = list_infeas.iloc[:, 0].tolist()
#        except:
#            print('Model cache file %s doesn\'t exist. '
#                  'Skipping.'%self.cache_fn)
#            list_infeas = None
#
#        if list_infeas:
#
#            self.df_comb = self.df_comb.loc[-self.df_comb.const_comb.isin(list_infeas)]

    def filter_invalid_solutions(self, cache=False):

        mask_empty = self.get_mask_empty_solution()

        #
        smaller_combs_inf = []
        for n in range(1, len(self.constrs_cols_neq)):

            combs = [c for c in map(set, itertools.combinations(self.constrs_cols_neq, n))
                     if not any(infc.issubset(c) for infc in smaller_combs_inf)]

            for comb in combs:
                val_combs = self.df_comb[comb].apply(tuple, axis=1).unique().tolist()

                for val_comb in val_combs:
                    mask = ~mask_empty.copy()
                    for val, col in list(zip(val_comb, comb)):
                        mask *= self.df_comb[col] == val

                    count_feas = len(self.df_comb.loc[mask, list(comb)].drop_duplicates())
                    if count_feas == 0:
                        print(comb, val_comb, count_feas)
                        smaller_combs_inf.append(comb)

        ncomb0 = len(self.df_comb)
        nempty = mask_empty.sum()
        shareempty = nempty / ncomb0 * 100
        logger.info('Number of empty solutions: '
                    '{:d} ({:.1f}%)'.format(nempty, shareempty))




        self.df_comb.loc[mask_empty, self.constrs_cols_neq].iloc[:, [0, 1, 2]].drop_duplicates()


        print('The following constraint combinations have empty solutions:\n',
              self.df_comb.loc[mask_empty, self.constrs_cols_neq])

        list_infeas = self.df_comb.loc[mask_empty, 'const_comb']

        self.df_comb = self.df_comb.loc[-mask_empty]

        mask_lindep = self.get_mask_linear_dependencies()

        nkey1, nkey2 = (mask_lindep == 1).sum(), (mask_lindep == 2).sum()
        logger.warning(('Number of solutions with linear dependencies: '
                       'Key 1: {:d} ({:.1f}%), Key 2: {:d} ({:.1f}%)'
                       ).format(nkey1, nkey1/ncomb0*100,
                                nkey2, nkey2/ncomb0*100))

        self.df_comb = pd.concat([self.df_comb, mask_lindep], axis=1)
        self.df_comb = self.df_comb.loc[-(mask_lindep > 1)]


        self.df_comb['result'] = \
                self.df_comb.apply(self.fix_linear_dependencies, axis=1)


        # arguably obsolete
#        if cache:
#            self.cache(list_infeas)


    def fix_linear_dependencies(self, x):
        '''
        All solutions showing linear dependencies are set to zero. See doc
        of symenergy.core.model.Model.get_mask_linear_dependencies
        '''

        if __name__ == '__main__':
            x = self.df_comb.iloc[1]

        if x.mask_res_unq == 0:
            list_res_new = list(list(x.result)[0])

        elif x.mask_res_unq == 1:

            list_res = list(x.result)[0]
            list_var = x.variabs_multips

            collect = {}

            list_res_new = [res for res in list_res]

            for nres, res in enumerate(list_res):

                free_symbs = [var for var in list_var if var in res.free_symbols]

                if free_symbs:
                    list_res_new[nres] = sp.numbers.Zero()
                    collect[list_var[nres]] = ', '.join(map(str, free_symbs))

            if collect:
                logger.info('%d'%x.idx)
                for res, var in collect.items():
                    logger.info('     Solution for %s contained variabs %s.'%(res, var))
        else:
            raise ValueError('mask_res_unq must be 0 or 1')

        return [list_res_new]

    @staticmethod
    def delete_cached(fn):
        if os.path.isfile(fn):
            logger.info('Removing file %s'%os.path.abspath(fn))
            os.remove(fn)
        else:
            logger.info('File doesn\'t exist. '
                        'Could not remove %s'%os.path.abspath(fn))

    def __repr__(self):

        ret = str(self.__class__)
        return ret

    def get_all_is_capacity_constrained(self):
        ''' Collects capacity constrained variables of all components. '''

        return [var
                for comp in self.comps.values()
                for var in comp.get_is_capacity_constrained()]

    def get_all_is_positive(self):
        ''' Collects positive variables of all components. '''

        is_positive_comps = [var
                             for comp in self.comps.values()
                             for var in comp.get_is_positive()]

        return is_positive_comps
