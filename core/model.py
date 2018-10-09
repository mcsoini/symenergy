#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the main Model class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import os

import itertools
import pandas as pd
import sympy as sp
import numpy as np
import multiprocessing
import hashlib

import symenergy
from symenergy.assets.plant import Plant
from symenergy.assets.storage import Storage
from symenergy.assets.curtailment import Curtailment
from symenergy.core.constraint import Constraint
from symenergy.core.slot import Slot, noneslot
from symenergy.core.parameter import Parameter


class Model:

    def __init__(self, curtailment=False):

        self.plants = {}
        self.slots = {}
        self.storages = {}

        self.comps = []

        # global vre scaling parameter, set to 1; used for evaluation
        self.vre_scale = Parameter('vre_scale', noneslot, 1)

        self.noneslot = noneslot

        self.curtailment = curtailment

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

            self.init_cache_filename()

        return wrapper

    def init_curtailment(self):

        if self.curtailment:
            self.curt = Curtailment('curt', self.slots)
            self.comps['curt'] = self.curt

    def init_cache_filename(self):

        fn = '%s.csv'%self.get_name()
        fn = os.path.join(list(symenergy.__path__)[0], 'cache_infeasible', fn)

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
        self.init_total_cost()

    @update_component_list
    def add_slot(self, name, *args, **kwargs):

        assert isinstance(name, str), 'Slot name must be string.'

        self.slots.update({name: Slot(name, **kwargs)})

    def init_total_param_values(self):
        '''
        Generates dictionary {parameter object: parameter value}
        for all parameters for all components.

        FOR NOW SLOTS ONLY HAVE PARAMS_TIME, PLANTS ONLY HAVE PARAMS_.
        '''

        self.param_values = {symb: val
                             for comp in self.comps.values()
                             for symb, val
                             in comp.get_params_dict(('symb',
                                                     'value')).items()}

        self.param_values.update({self.vre_scale.symb: self.vre_scale.value})

    def init_total_cost(self):

        self.tc = sum(p.cc for p in self.plants.values())

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

        print(self.cstr_supply)

    def get_supply_constraint_expr(self, cstr):
        '''
        Initialize the load constraints for a given time slot.
        Note: this accesses all plants, therefore method of the model class.
        '''

        slot = cstr.slot

        total_chg = sum(p
                     for store in self.storages.values()
                     for slt, p in store.get_chgdch('chg').items()
                     if slt in store.get_chgdch('chg').keys()
                     and slt == slot)
        total_dch = sum(p
                     for store in self.storages.values()
                     for slt, p in store.get_chgdch('dch').items()
                     if slt in store.get_chgdch('dch').keys()
                     and slt == slot)

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

    def collect_component_constraints(self):
        '''
        Note: Doesn't include supply constraints, which are a model attribute.
        '''

        self.constrs = {}
        for comp in self.comps.values():
            for key, attr in comp.__dict__.items():
                if key.startswith('cstr_'):
                    for slot, cstr in attr.items():
                        self.constrs[cstr] = comp

        # dictionary {column name: constraint object}
        self.constrs_dict = {cstr.col: cstr for cstr in self.constrs.keys()}

        # list of constraint columns
        self.constrs_cols = [cstr.col for cstr in self.constrs.keys()]

    def init_constraint_combinations(self):
        '''
        Gathers all non-equal component constraints,
        calculates cross-product of all
        (binding, non-binding) combinations and instantiates dataframe.
        '''

        self.constrs_neq = [cstr for cstr in self.constrs
                            if not cstr.is_equality_constraint]
        self.constrs_cols_neq = [cstr.col for cstr
                                 in self.constrs_neq]

        list_combs = list(itertools.product(*[[0, 1] for cc
                                              in self.constrs_neq]))
        self.df_comb = pd.DataFrame(list_combs, columns=self.constrs_cols_neq)

        self.df_comb = self.combine_constraint_names(self.df_comb)



    def remove_mutually_exclusive_combinations(self):

        # [(cstr_pattern1, cstr_pattern2, is_exclusive), (...)]
        multips_mut_excl = [('pos_p', 'cap_C', True),
                            ('cap_E', 'pos_e', True)]

        cols_mut_excl_0 = []

        for name_cstr1, name_cstr2, is_exclusive in multips_mut_excl:

            for cstr in [cstr for cstr in self.constrs_cols_neq
                         if name_cstr1 in cstr]:

                cstr_pattern = cstr.replace(name_cstr1, '')

                select_cstr = [cstr_match
                               for cstr_match in self.constrs_cols_neq
                               if name_cstr2 in cstr_match
                               and cstr_pattern == cstr_match.replace(name_cstr2, '')
                               ]
                if len(select_cstr) > 1:
                    raise RuntimeError(
                            'remove_mutually_exclusive_combinations: '
                            'select_cstr for (%s, %s) has length > 1: '
                            %(name_cstr1, name_cstr2) + ''
                            '%s'%str(select_cstr))
                elif len(select_cstr) == 1:
                    select_cstr = select_cstr[0]
                    cols_mut_excl_0.append((cstr, select_cstr, is_exclusive))

        cols_mut_excl_0 += [('act_lb_phs_pos_p_day', 'act_lb_phs_pos_p_evening', False),
                          ('act_lb_phs_pos_p_day', 'act_lb_phs_pos_e_None', False),
                          ('act_lb_phs_pos_p_evening', 'act_lb_phs_pos_e_None', False)]

        # make sure all cols are present
        cols_mut_excl = []
        for comb in cols_mut_excl_0:
            print([col in self.df_comb.columns or isinstance(col, bool)
                    for col in comb])

            if all([col in self.df_comb.columns or isinstance(col, bool)
                        for col in comb]):
                cols_mut_excl.append(comb)


        print('Deleting mutually exclusive constraints from '
              'df_comb (%s rows)'%len(self.df_comb), end=' ... ')
        tot_delete = 0

#        col1, col2, is_exclusive = cols_mut_excl[2]
        for col1, col2, is_exclusive in cols_mut_excl:

            if is_exclusive:
                mask = ((self.df_comb[col1] == 1)
                      & (self.df_comb[col2] == 1))
                self.df_comb = self.df_comb.loc[-mask]
            else:
                mask = (((self.df_comb[col1] == 1) & (self.df_comb[col2] == 1))
                      | ((self.df_comb[col1] == 0) & (self.df_comb[col2] == 0)))
                self.df_comb = self.df_comb.loc[mask]
            print(self.df_comb.loc[mask])
            tot_delete += mask.sum()
        print('%d rows deleted, %s remaining.'%(tot_delete,
                                                 len(self.df_comb)))

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

    def construct_lagrange(self, row):

        slct_constr = row.loc[self.constrs_cols_neq].to_dict()

        lagrange = self.tc
        lagrange += sum([cstr.expr for cstr in self.cstr_supply])
        lagrange += sum([cstr.expr for cstr in self.constrs
                         if cstr.is_equality_constraint])


        constr, is_active = list(slct_constr.items())[0]
        for col, is_active in slct_constr.items():

            if is_active:

                cstr = self.constrs_dict[col]

                lagrange += cstr.expr

        return lagrange


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
        variabs_slct = [ss for ss in lfs if ss in self.variabs.keys()]
        multips_slct = [ss for ss in lfs if ss in self.multips.keys()]

        return variabs_slct + multips_slct

    def solve(self, lagrange, variabs_multips_slct, index):

        if __name__ == '__main__':
            slct_index = 20
            lagrange = self.df_comb.iloc[slct_index].lagrange
            variabs_multips_slct = self.df_comb.iloc[slct_index].variabs_multips

        mat = sp.tensor.array.derive_by_array(lagrange, variabs_multips_slct)
        mat = sp.Matrix(mat).expand()

        A, b = sp.linear_eq_to_matrix(mat, variabs_multips_slct)

        self.print_row(index, (A.rows, A.cols))

        solution = sp.linsolve((A, b), variabs_multips_slct)

        return solution

    def print_row(self, index, matrix_dim):

        matrix_dim = 'Matrix %s'%'x'.join(map(str, matrix_dim))

        strg = 'Constr. comb. %d out of %d: %s.'%(index, self.n_comb,
                                                     matrix_dim)

        print(strg)


    def call_solve_df(self, df):
        ''' Applies to dataframe. '''

        slct_cols = ['lagrange', 'variabs_multips', 'idx']

        return df[slct_cols].apply(lambda x: self.solve(x.lagrange,
                                                        x.variabs_multips,
                                                        x.idx), axis=1)

    def solve_all(self, nthreads=None):

        self.define_problems()

        nthreads = min(nthreads, len(self.df_comb))
        nchunks = min(nthreads * 4, len(self.df_comb))

        def parallelize_df(df, func):
            df_split = np.array_split(df, nchunks)
            pool = multiprocessing.Pool(nthreads)
            df = pd.concat(pool.map(func, df_split))
            pool.close()
            pool.join()
            return df


        if not nthreads:
            self.df_comb['result'] = self.call_solve_df(self.df_comb)

        else:
            self.df_comb['result'] = parallelize_df(self.df_comb,
                                                    self.call_solve_df)


        call_subs_tc = lambda x: self.subs_total_cost(x.result,
                                                      x.variabs_multips)

        self.df_comb['tc'] = self.df_comb.apply(call_subs_tc, axis=1)

    def combine_constraint_names(self, df):

        constr_name = pd.DataFrame(index=df.index)
        for const in self.constrs_cols_neq:

            constr_name[const] = const + '=' + df[const].astype(str)

        join = lambda x: ', '.join(x)
        df['const_comb'] = constr_name.apply(join, axis=1)

        return df


    def define_problems(self):
        '''
        For each combination of constraints, get the lagrangian
        and the variables.
        '''

        print('Defining lagrangians...')
        self.df_comb['lagrange'] = \
            self.df_comb.apply(self.construct_lagrange, axis=1)

        print('Getting selected variables/multipliers...')
        self.df_comb['variabs_multips'] = \
            self.df_comb.lagrange.apply(self.get_variabs_multips_slct)

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

        check_emptyset = lambda res: isinstance(res, sp.sets.EmptySet)
        mask_empty = self.df_comb.result.apply(check_emptyset)

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
        res_vars.loc[mask_valid, 'res_comps'] = res_vars.loc[mask_valid].apply(get_comps, axis=1)

        # maximum unique component number
        get_max_nunq = lambda x: max(len(set(comp_list))
                                     for comp_list in x.res_comps) if x.res_comps else 0
        res_vars.loc[mask_valid, 'mask_res_unq'] = res_vars.loc[mask_valid].apply(get_max_nunq, axis=1)

        return res_vars.mask_res_unq



    def get_name(self):
        '''
        Returns a unique hashed model name based on the constraint names.
        '''
        m_name = '_'.join([cstr.base_name for cstr in self.constrs])

        m_name = hashlib.md5(m_name.encode('utf-8')).hexdigest()[:12].upper()

        return m_name

    def cache(self, list_infeas):
        '''
        Saves list of infeasible constraint combinations to file on disk.
        '''

        list_infeas.to_csv(self.cache_fn, index=False)

    def delete_cached(self):

        try:
            list_infeas = pd.read_csv(self.cache_fn, header=None)
            list_infeas = list_infeas.iloc[:, 0].tolist()
        except:
            print('Model cache file %s doesn\'t exist. '
                  'Skipping.'%self.cache_fn)
            list_infeas = None

        if list_infeas:

            self.df_comb = self.df_comb.loc[-self.df_comb.const_comb.isin(list_infeas)]



    def filter_invalid_solutions(self, cache=True):

        mask_empty = self.get_mask_empty_solution()

        print('The following constraint combinations have empty solutions:\n',
              self.df_comb.loc[mask_empty, self.constrs_cols_neq])

        list_infeas = self.df_comb.loc[mask_empty, 'const_comb']

        self.df_comb = self.df_comb.loc[-mask_empty]

        mask_lindep = self.get_mask_linear_dependencies()

        print('The following constraint combinations '
              'have mixed residual interdependencies of variables '
              'and are removed:\n',
              self.df_comb.loc[(mask_lindep > 1), self.constrs_cols_neq])

        list_infeas = pd.concat([list_infeas,
                        self.df_comb.loc[mask_lindep > 1,
                                          'const_comb']], axis=0)

        self.df_comb = pd.concat([self.df_comb, mask_lindep], axis=1)
        self.df_comb = self.df_comb.loc[-(mask_lindep > 1)]


        self.df_comb['result'] = \
                self.df_comb.apply(self.fix_linear_dependencies, axis=1)

        if cache:
            self.cache(list_infeas)


    def fix_linear_dependencies(self, x):
        '''
        Linear solutions

        All solutions showing linear dependencies are set to zero.
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
                print('%s'%x[self.constrs_cols_neq])
                for res, var in collect.items():
                    print('     Solution for %s contained variabs %s.'%(res, var))

        return [list_res_new]

    def subs_total_cost(self, result, var_mlt_slct):
        '''
        Substitutes solution into TC variables.
        This expresses the total cost as a function of the parameters.
        '''

        dict_var = {var: list(result)[0][ivar]
                    if not isinstance(result, sp.sets.EmptySet)
                    else np.nan for ivar, var
                    in enumerate(var_mlt_slct)}

        return self.tc.copy().subs(dict_var)

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
