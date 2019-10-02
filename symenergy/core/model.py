#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the main Model class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import sys

from pathlib import Path
import itertools
import pandas as pd
import sympy as sp
import numpy as np
import time
from hashlib import md5
from sympy.tensor.array import derive_by_array

from symenergy.assets.plant import Plant
from symenergy.assets.storage import Storage
from symenergy.assets.curtailment import Curtailment
from symenergy.core.constraint import Constraint
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter
from symenergy.auxiliary.parallelization import parallelize_df
from symenergy.auxiliary.parallelization import MP_COUNTER, MP_EMA
from symenergy.auxiliary.parallelization import log_time_progress
from symenergy import _get_logger
from symenergy.patches.sympy_linsolve import linsolve
from symenergy.auxiliary.constrcomb import filter_constraint_combinations
from symenergy.auxiliary import io

logger = _get_logger(__name__)

logger.warning('!!! Monkey-patching sympy.linsolve !!!')
sp.linsolve = linsolve


if __name__ == '__main__': sys.exit()


class Model:



    _MUTUALLY_EXCLUSIVE = {
        'No power production when curtailing':
                (('pos_p', 'this', False), ('curt_pos_p', 'this', False)),
        'No discharging when curtailing':
                (('pos_pdch', 'this', False), ('curt_pos_p', 'this', False))
         }

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

            self.cache = io.Cache(self.get_model_hash_name())

        return wrapper

    def init_curtailment(self):

        if self.curtailment:
            self.curt = Curtailment('curt', self.slots)
            self.comps['curt'] = self.curt

    @property
    def df_comb(self):
        return self._df_comb

    @df_comb.setter
    def df_comb(self, df_comb):
        self._df_comb = df_comb.reset_index(drop=True)
        if self.nress:
            self.nress = len(self._df_comb)
        else:
            self.ncomb = len(self._df_comb)


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

        self.tc = sum(p.cc for p in
                      list(self.plants.values())
                           + list(self.storages.values()))
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

        equ = (slot.l.symb
               - slot.vre.symb * self.vre_scale.symb
               + total_chg
               - total_dch
               - sum(plant.p[slot] for plant
                     in self.plants.values()))

        if self.curtailment:
            equ += self.curt.p[slot]

        return equ

    def generate_solve(self):

        if self.cache.file_exists:
            self.df_comb = self.cache.load()
        else:
            self.init_constraint_combinations()
            self.define_problems()
            self.solve_all()
            self.filter_invalid_solutions()
            self.generate_total_costs()
#            self.fix_stored_energy()
            self.cache.write(self.df_comb)


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


#
#    def _fix_stored_energy_full(self, x):
##
##        x = self.df_comb.loc[]
#
#        dict_var = self.get_result_dict(x, True)
##
##        store = self.comps['phs']
##
##        store.pchg
#


    def _fix_stored_energy(self, x):
        '''
        Recalculates stored energy from power results.

        Input parameters:
            * x -- df_comb table row, must contain columns
                   'variabs_multibs', 'result'
            * name -- storage object name

        TODO: Should be done by storage class.
        TODO: Doesn't work for free storage (not slots_map constraint).
        '''

        dict_var = dict(zip(map(str, x.variabs_multips), list(x.result)[0]))

        for store in self.storages.values():
            sum_chg = sum(dict_var['%s_pchg_%s'%(store.name, chg_slot)]
                          * (self.slots[chg_slot].weight
                             if self.slots[chg_slot].weight else 1)
                          for chg_slot in
                          tuple(slots for slots in store.slots_map['chg']))

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
        self.constrs_cols_neq = [cstr.col for cstr in self.constrs
                                 if not cstr.is_equality_constraint]


    def get_model_mutually_exclusive_cols(self):
        '''
        Expand model MUTUALLY_EXCLUSIVE to plants and time slots.

        The initial list of constraint combinations is filtered only according
        to constraint combinations within each component separately. Here,
        additional constraint combinations from different components are
        removed.

        Assuming only `'this'` as slottype.

        TODO: Integrate with constrcomb.CstrCombBase or derived class thereof.
        '''

        list_col_names = []
        for mename, me in self._MUTUALLY_EXCLUSIVE.items():

            # identify relevant constraints of all components
            cstrs_all = [comp.get_constraints(False, True, True)
                         for comp in self.comps.values()]
            cstrs_all = dict(pair for d in cstrs_all for pair in d.items())
            cstrs_all = {(key[0], '{}_{}'.format(key[0],
                                        key[1].replace('cstr_', ''))): val
                         for key, val in cstrs_all.items()}

            # expand to all components
            me_exp = [tuple((cstrs, name_cstr[0], me_slct[-1])
                       for name_cstr, cstrs in cstrs_all.items()
                       if name_cstr[1].endswith(me_slct[0]))
                      for me_slct in me]
            # all components of the combination's two constraints
            me_exp = list(itertools.product(*me_exp))

            # remove double components, also: remove component names
            me_exp = [tuple((cstr[0], cstr[2]) for cstr in cstrs)
                      for cstrs in me_exp
                      if not cstrs[0][1] == cstrs[1][1]]

            me_exp = [tuple({slot: (cstr, cstrs[1])
                                  for slot, cstr in cstrs[0].items()}
                            for cstrs in cstr_comb)
                      for cstr_comb in me_exp]

            # split by time slots for existing time slots
            me_exp = [(cstr_comb[0][slot], cstr_comb[1][slot])
                      for cstr_comb in me_exp
                      for slot in self.slots.values()
                      if all(slot in cc for cc in cstr_comb)]

            # switch from constraint objects to column names
            me_exp = [tuple((cstr[0].col, cstr[1])
                      for cstr in cstrs) for cstrs in me_exp]

            list_col_names += me_exp

        return list_col_names


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

        logger.info('Length of merged df_comb: %d'%len(dfcomb))

        # filter according to model _MUTUALLY_EXCLUSIVE
        logger.info('*'*30 + 'model filtering' + '*'*30)
        model_mut_excl_cols = self.get_model_mutually_exclusive_cols()
        dfcomb = filter_constraint_combinations(dfcomb, model_mut_excl_cols)

        self.df_comb = dfcomb.drop('dummy', axis=1)
        self.ncomb = len(self.df_comb)



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



# =============================================================================
#     Various solver-related methods
# =============================================================================

    def solve(self, x):

        # substitute variables with binding positivitiy constraints
        subs_zero = {cstr.expr_0: sp.Integer(0) for col, cstr
                     in self.constrs_dict.items()
                     if cstr.is_positivity_constraint
                     and x[cstr.col]}

        mat = derive_by_array(x.lagrange, x.variabs_multips)
        mat = sp.Matrix(mat).expand()
        mat = mat.subs(subs_zero)

        variabs_multips_slct = list(set(x.variabs_multips) - set(subs_zero))

        A, b = sp.linear_eq_to_matrix(mat, variabs_multips_slct)

        MP_COUNTER.increment()
        solution_0 = sp.linsolve((A, b), variabs_multips_slct)

        if isinstance(solution_0, sp.sets.EmptySet):
            return None

        else:

            # init with zeros
            solution_dict = dict.fromkeys(x.variabs_multips, sp.Integer(0))

            # update with solutions
            solution_dict.update(dict(zip(variabs_multips_slct,
                                          list(solution_0)[0])))

            solution = tuple(solution_dict.values())


            return solution


    def wrapper_call_solve_df(self, df, *args):

        name, ntot = 'Solve', self.ncomb
        return log_time_progress(self.call_solve_df)(self, df, name, ntot)


    def call_solve_df(self, df):
        ''' Applies to dataframe. '''

        return df.apply(self.solve, axis=1).tolist()


    def solve_all(self):

        logger.info('Solving')

        if __name__ == '__main__':
            lagrange, variabs_multips_slct, index = self.df_comb[0]
            x = self.df_comb.iloc[0]

        if not self.nthreads:
            self.df_comb['result'] = self.call_solve_df(self.df_comb)
        else:
            func = self.wrapper_call_solve_df
            self.df_comb['result'] = parallelize_df(self.df_comb,
                                                    func, self.nthreads)

# =============================================================================
# =============================================================================

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
            func = self.wrapper_call_subs_tc
            self.df_comb['tc'] = parallelize_df(df, func, self.nthreads)

    def subs_total_cost(self, res, var, idx):
        '''
        Substitutes solution into TC variables.
        This expresses the total cost as a function of the parameters.
        '''

        dict_var = {var: res[ivar]
                    if not isinstance(res, sp.sets.EmptySet)
                    else np.nan for ivar, var
                    in enumerate(var)}

        MP_COUNTER.increment()

        return self.tc.copy().subs(dict_var)


    def call_subs_tc(self, df):

        return [self.subs_total_cost(res, var, idx) for res, var, idx in df]

    def wrapper_call_subs_tc(self, df, *args):

        name = 'Substituting total cost'
        ntot = self.nress
        return log_time_progress(self.call_subs_tc)(self, df, name, ntot)

# =============================================================================
# =============================================================================


    def construct_lagrange(self, row):

        lagrange = self.lagrange_0

        active_cstrs = row[row == 1].index.values
        lagrange += sum(self.constrs_dict[cstr_name].expr
                        for cstr_name in active_cstrs)
        MP_COUNTER.increment()

        return lagrange


    def call_construct_lagrange(self, df):
        '''
        Top-level method for parallelization of construct_lagrange.
        '''

        return df.apply(self.construct_lagrange, axis=1).tolist()

    def wrapper_call_construct_lagrange(self, df, *args):

        name = 'Construct lagrange'
        ntot = self.ncomb
        return log_time_progress(self.call_construct_lagrange)(self, df, name, ntot)


# =============================================================================
# =============================================================================

    def get_variabs_multips_slct(self, lagrange):
        '''
        Returns all relevant variables and multipliers fcall_subs_tcor this model.

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
        MP_COUNTER.increment()
        return [ss for ss in lfs if ss in self.variabs or ss in self.multips]


    def call_get_variabs_multips_slct(self, df):

        res = list(map(self.get_variabs_multips_slct, df))

        return res

    def wrapper_call_get_variabs_multips_slct(self, df, *args):

        name = 'Get variabs/multipliers'
        ntot = self.ncomb
        func = self.call_get_variabs_multips_slct
        return log_time_progress(func)(self, df, name, ntot)

# =============================================================================
# =============================================================================


    def define_problems(self):
        '''
        For each combination of constraints, get the lagrangian
        and the variables.
        '''

        logger.info('Defining lagrangians...')
        df = self.df_comb[self.constrs_cols_neq]
        if not self.nthreads:
            self.df_comb['lagrange'] = self.call_construct_lagrange(df)
        else:
            func = self.wrapper_call_construct_lagrange
            nthreads = self.nthreads
            self.df_comb['lagrange'] = parallelize_df(df, func, nthreads)

        logger.info('Getting selected variables/multipliers...')
        df = self.df_comb.lagrange
        if not self.nthreads:
            self.list_variabs_multips = self.call_get_variabs_multips_slct(df)
            self.df_comb['variabs_multips'] = self.list_variabs_multips
        else:
            func = self.wrapper_call_get_variabs_multips_slct
            nthreads = self.nthreads
            self.df_comb['variabs_multips'] = parallelize_df(df, func, nthreads)

        # get index
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


#    def combine_constraint_names(self, df):
#
#        constr_name = pd.DataFrame(index=df.index)
#        for const in self.constrs_cols_neq:
#
#            constr_name[const] = const + '=' + df[const].astype(str)
#
#        join = lambda x: ', '.join(x)
#        df['const_comb'] = constr_name.apply(join, axis=1)
#
#        return df


    def get_mask_linear_dependencies(self):
        '''
        Solutions of problems containing linear dependencies.

        In case of linear dependencies SymPy returns solutions containing
        variables which we are actually solving for. To fix this, we
        differentiate between two cases:


        1. All corresponding solutions belong to the same component. Overspecification
           occurs if the variables of the same component
           depend on each other but are all zero. E.g. charging,
           discharging, and stored energy in the case of storage.
           They are set to zero.
        2. Linear dependent variables belonging to different components.
           This occurs if the model is underspecified, e.g. if it doesn't
           matter which component power is used. Then the solution can
           be discarded without loss of generality. All cases will still
           be captured by other constraint combinations.
        3. Different components but same component classes. If multiple idling
           storage plants are present, their mulitpliers show linear
           dependencies.

        Returns:
            * Series mask with values: 0 -> no variables in solutions; 1 ->
                single variable in solution: set to zero; > 1 -> mixed
                interdependency: drop solution; NaN: empty solution set
        '''

#        check_vars_left = lambda x: any(var in x.result.free_symbols
#                                        for var in x.variabs_multips)
#        mask_lindep = self.df_comb.apply(check_vars_left, axis=1)

        # get residual variables
        if __name__ == '__main__':
            x = self.df_comb.iloc[0]

        # mask with non-empty solutions
        not_empty = lambda x: not isinstance(x, sp.sets.EmptySet)
        mask_valid = self.df_comb.result.apply(not_empty)

        # for each individual solution, get residual variables/multipliers
        def get_residual_vars(x):
            return tuple([var for var in res.free_symbols
                                 if var in x.variabs_multips]
                            for nres, res in enumerate(x.result))

        res_vars = self.df_comb[['result', 'variabs_multips', 'idx']].copy()
        res_vars.loc[mask_valid, 'res_vars'] = \
                self.df_comb.loc[mask_valid].apply(get_residual_vars, axis=1)

        if __name__ == '__main__':
            x = res_vars.iloc[1]

        # add solution variable itself to all non-empty lists
        def add_solution_var(x):
            return tuple(x.res_vars[nres_vars]
                                + [x.variabs_multips[nres_vars]]
                            if res_vars else []
                            for nres_vars, res_vars in enumerate(x.res_vars))

        res_vars.loc[mask_valid, 'res_vars'] = \
                res_vars.loc[mask_valid].apply(add_solution_var, axis=1)

        # get component corresponding to variable
        comp_dict_varmtp = self.multips.copy()
        comp_dict_varmtp.update(self.variabs)

        def get_comps(x):
            return tuple((list(set([comp_dict_varmtp[var]
                                              for var in res_vars])))
                                    for res_vars in x.res_vars
                                    if res_vars)
        res_vars.loc[mask_valid, 'res_comps'] = \
                res_vars.loc[mask_valid].apply(get_comps, axis=1)

        # maximum unique component number
        def get_max_ncompunq(x):
            nmax = 0
            if x.res_comps:
                nmax = max(len(set(comp_list)) for comp_list in x.res_comps)
            return nmax
        res_vars.loc[mask_valid, 'ncompunq'] = \
                res_vars.loc[mask_valid].apply(get_max_ncompunq, axis=1)

        # maximum unique component class number
        def get_max_ncompclassunq(x):
            nmax = 0
            if x.res_comps:
                nmax = len(set([x.__class__ for x in
                                itertools.chain.from_iterable(x.res_comps)]))
            return nmax
        res_vars.loc[mask_valid, 'ncompclassesunq'] = \
                res_vars.loc[mask_valid].apply(get_max_ncompclassunq, axis=1)

        # generate lindep codes
        res_vars['code_lindep'] = 0


        res_vars.loc[res_vars.ncompunq < 2, 'code_lindep'] = 1
        res_vars.loc[(res_vars.ncompunq >= 2)
                     & (res_vars.ncompclassesunq >= 2 ), 'code_lindep'] = 2
        res_vars.loc[(res_vars.ncompunq >= 2)
                     & (res_vars.ncompclassesunq < 2 ), 'code_lindep'] = 3

        return res_vars.code_lindep


    def filter_invalid_solutions(self):
        '''
        '''


#        #
#        smaller_combs_inf = []
#        for n in range(1, len(self.constrs_cols_neq)):
#
#            combs = [c for c in map(set, itertools.combinations(self.constrs_cols_neq, n))
#                     if not any(infc.issubset(c) for infc in smaller_combs_inf)]
#
#            for comb in combs:
#                val_combs = self.df_comb[comb].apply(tuple, axis=1).unique().tolist()
#
#                for val_comb in val_combs:
#                    mask = ~mask_empty.copy()
#                    for val, col in list(zip(val_comb, comb)):
#                        mask *= self.df_comb[col] == val
#
#                    count_feas = len(self.df_comb.loc[mask, list(comb)].drop_duplicates())
#                    if count_feas == 0:
#                        print(comb, val_comb, count_feas)
#                        smaller_combs_inf.append(comb)
#
#
#
#
#
#        self.df_comb.loc[mask_empty, self.constrs_cols_neq].iloc[:, [0, 1, 2]].drop_duplicates()
#
#
#        print('The following constraint combinations have empty solutions:\n',
#              self.df_comb.loc[mask_empty, self.constrs_cols_neq])
#
#        list_infeas = self.df_comb.loc[mask_empty, 'const_comb']


        mask_empty = self.get_mask_empty_solution()

        ncomb0 = len(self.df_comb)
        nempty = mask_empty.sum()
        shareempty = nempty / ncomb0 * 100
        logger.info('Number of empty solutions: '
                    '{:d} ({:.1f}%)'.format(nempty, shareempty))

        # keep empty solutions constraint combinations for post-analysis
        self.df_comb_invalid = self.df_comb.loc[mask_empty,
                                                self.constrs_cols_neq]

        # remove invalid constraint combinations
        self.df_comb = self.df_comb.loc[-mask_empty]

        # get info on linear combinations
        mask_lindep = self.get_mask_linear_dependencies()

        ncomb0 = len(self.df_comb)
        nkey1, nkey2, nkey3 = (mask_lindep == 1).sum(), (mask_lindep == 2).sum(), (mask_lindep == 3).sum()
        logger.warning(('Number of solutions with linear dependencies: '
                       'Key 1: {:d} ({:.1f}%), Key 2: {:d} ({:.1f}%), Key 3: {:d} ({:.1f}%)'
                       ).format(nkey1, nkey1/ncomb0*100,
                                nkey2, nkey2/ncomb0*100,
                                nkey3, nkey3/ncomb0*100))

        self.df_comb = pd.concat([self.df_comb, mask_lindep], axis=1)
        self.df_comb = self.df_comb.loc[-(self.df_comb.code_lindep == 2)]

        # adjust results for single-component linear dependencies
        self.df_comb['result'] = \
                self.df_comb.apply(self.fix_linear_dependencies, axis=1)

        self.nress = len(self.df_comb)


    def fix_linear_dependencies(self, x):
        '''
        All solutions showing linear dependencies are set to zero. See doc
        of symenergy.core.model.Model.get_mask_linear_dependencies
        '''

        if __name__ == '__main__':
            x = self.df_comb.iloc[1]

        if x.code_lindep == 0:
            list_res_new = x.result

        if x.code_lindep == 3:
            list_res_new = x.result

        elif x.code_lindep == 1:

            list_res = x.result
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
            raise ValueError('code_lindep must be 0, 3, or 1')

        return list_res_new

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


    def print_mutually_exclusive_post(self, logging=False):

        print_func = logger.info if logging else print

        dfiv = self.df_comb_invalid
        dfvl = self.df_comb[self.df_comb_invalid.columns]

        tot_list_excl = []

        ncols = 3
        for ncols in range(2, len(dfvl.columns)):
            print_func('ncols=%d'%ncols)
            for slct_cols in tuple(itertools.combinations(dfvl.columns, ncols)):

                get_combs = lambda df: (df[list(slct_cols)].drop_duplicates()
                                                .apply(tuple, axis=1).tolist())
                vals_dfiv_slct = get_combs(dfiv)
                vals_dfvl_slct = get_combs(dfvl)

                # any in dfvl
                vals_remain = [comb for comb in vals_dfiv_slct
                               if not comb in vals_dfvl_slct]

                if vals_remain:
                    list_exc = [tuple(zip(*colvals)) for colvals
                                in list(zip([slct_cols] * ncols, vals_remain))]

                    # check not superset of tot_list_excl elements
                    list_exc = [comb  for comb in list_exc
                                 if not any(set(comb_old).issubset(set(comb))
                                 for comb_old in tot_list_excl)]

                    if list_exc:
                        print_func(list_exc)

                    tot_list_excl += list_exc

    def draw_slots(self, graphwidth=70):

        slotlist = [(slotname, slot.l.value, slot.vre.value)
                    for slotname, slot in self.slots.items()]
        maxlen = len(max([slot[0] for slot in slotlist], key=len))
        maxpwr = max(itertools.chain.from_iterable(slot[1:]
                                                   for slot in slotlist))

        ljust_all = lambda lst, newlen: [(x[0].ljust(newlen),) + x[1:]
                                         for x in lst]

        slotlist = ljust_all(slotlist, maxlen + 1)
        slotlist = [(slotname,
                     round(l / maxpwr * graphwidth),
                     round(vre / maxpwr * graphwidth))
                    for slotname, l, vre in slotlist]
        bar = lambda l, vre: ((l - vre) * "\u2588"  + vre * "\u2591")
        slotlist = [(name, bar(l, vre).ljust(graphwidth))
                    for name, l, vre in slotlist]

        for slotbar, slotobj in zip(slotlist, self.slots.values()):
            slot, bar = slotbar
            data = 'l={:.1f}/vre={:.1f}'.format(slotobj.l.value, slotobj.vre.value)
            print(slot, bar, data, sep=' | ', end='\n', flush=True)

    def get_model_hash_name(self):

        hash_input = ''.join(comp.get_component_hash_name()
                             for comp in self.comps.values())
        return md5(hash_input.encode('utf-8')).hexdigest()
