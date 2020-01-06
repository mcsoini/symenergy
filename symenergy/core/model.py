#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the main Model class.

Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""
import sys

from pathlib import Path
import itertools
from collections import Counter
from orderedset import OrderedSet
import pandas as pd
import sympy as sp
import wrapt
import numpy as np
import textwrap
from sympy.tensor.array import derive_by_array

from symenergy.patches.symenergy_solveset import linear_eq_to_matrix
from symenergy.assets.plant import Plant
from symenergy.assets.storage import Storage
from symenergy.assets.curtailment import Curtailment
from symenergy.core.constraint import Constraint
from symenergy.core.slot import Slot, SlotBlock, noneslot
from symenergy.core.parameter import Parameter
from symenergy.auxiliary.parallelization import parallelize_df
from symenergy.auxiliary.parallelization import MP_COUNTER, MP_EMA
from symenergy.auxiliary.parallelization import log_time_progress
from symenergy.auxiliary.decorators import hexdigest
from symenergy import _get_logger
from symenergy.patches.sympy_linsolve import linsolve
#from symenergy.patches.sympy_linear_coeffs import linear_coeffs
from symenergy.auxiliary.constrcomb import filter_constraint_combinations
from symenergy.auxiliary import io


logger = _get_logger(__name__)

sp.linsolve = linsolve  # Monkey-patching sympy.linsolve

if __name__ == '__main__': sys.exit()

class Model:
    '''
    Instantiate a model object. Start from here.

    Parameters
    ----------
    slot_weights : float
        default time slot weight (hours); this instantiates a singleton
        parameter to avoid the definition of individual parameter for each
        slot; it can be overwritten for individual time slots if the weight
        parameter is provided
    constraint_filt : str
        see :func:`symenergy.core.model.Model.init_constraint_combinations`
    curtailment : bool or list of Slots, default False
        Allow for curtailment in each time slot (True) or in a selection of
        time slots. This generates a
        :class:`symenergy.assets.curtailment.Curtailment` instance `curt`,
        which defines the positive curtailment power variables `curt.p`
        for each of the relevant time slots.
    nworkers : int or False
        number of workers to be used for parallel model setup and solving;
        passed to the :class:`multiprocessing.Pool` initializer;
        defaults to `multiprocessing.cpu_count() - 1`;
        if False, no multiprocessing is used
    '''

    mutually_exclusive = {
        'No power production when curtailing':
                (('pos_p', 'this', False), ('curt_pos_p', 'this', False)),
        'No discharging when curtailing':
                (('pos_pdch', 'this', False), ('curt_pos_p', 'this', False))
         }

    def __init__(self, nworkers='default', curtailment=False,
                 slot_weight=1, constraint_filt=''):

        self.plants = {}
        self.slots = {}
        self.slot_blocks = {}
        self.storages = {}
        self.comps = {}
        self.curt = {}

        self.nress = None
        self.ncomb = None

        self._cache = None

        self.df_comb = None
        self.df_comb_invalid = None

        self.nworkers = nworkers
        self.constraint_filt = constraint_filt

        self._slot_weights = Parameter('w', noneslot, slot_weight)

        # global vre scaling parameter, set to 1; used for evaluation
        self.vre_scale = Parameter('vre_scale', noneslot, 1)

        self.noneslot = noneslot

        self.curtailment = curtailment

        self.ncomb = None  # determined through construction of self.df_comb
        self.nress = None  # number of valid results


    @wrapt.decorator
    def _update_component_list(f, self, args, kwargs):
        '''
        Rebuild all derived model attributes.

        This is triggered (decorator) every a relevant change is made to
        the model through the public API.
        '''
        f(*args, **kwargs)

        self.comps = {}
        self.comps.update(self.plants)
        self.comps.update(self.slots)
        self.comps.update(self.storages)
        self.comps.update(self.slot_blocks)

        if not 'curt' in self.comps and self.curtailment:
            logger.debug('Auto-adding curtailment')
            self._add_curtailment(self.slots)

        self.comps.update(self.curt)

        # aggregate all attribute collections
        self.parameters = sum(c.parameters._copy() for c in self.comps.values())
        self.parameters.append(self.vre_scale)
        self.variables = sum(c.variables._copy() for c in self.comps.values())
        self.constraints = sum(c.constraints._copy()
                               for c in self.comps.values())

        self._init_supply_constraints()

        self._init_total_cost()

        self.cache = io.Cache(self.get_model_hash_name())

#        self._assert_slot_block_validity()

        self.constrs_cols_neq = self.constraints.tolist('col',
                                            is_equality_constraint=False)

    @wrapt.decorator
    def _check_component_replacement(f, self, args, kwargs):

        if 'name' in kwargs:
            name = kwargs['name']
        elif args and isinstance(args[0], str):
            name = args[0]
        else:
            raise AttributeError('Name required for all components.')

        assert name  not in {**self.comps, **self.slot_blocks}, (
                'A component or slot_block `%s` has already been '
                         'defined.')%name

        return f(*args, **kwargs)

    @property
    def cache(self):
        if self._cache is None:
            raise AttributeError('Model cache not yet initialized. Please add '
                                 'a plant or storage asset.')

        return self._cache

    @cache.setter
    def cache(self, cache):
        self._cache = cache

    @_update_component_list
    def freeze_parameters(self, exceptions=None):
        '''
        Switch from variable to numerical value for all model parameters.
        This automatically updates the definition of all sympy expressions
        (total cost, lagrange, ...)

        Parameters
        ----------
        exceptions : list(str)
            names of parameters to be excluded; corresponds to elements of the
            list `Model.parameters('name')`


        Example
        -------
        .. code-block:: python

            >>> from symenery.core import model
            >>> m = model.Model()
            >>> m.add_slot('s0', load=2, vre=3)
            >>> m.add_plant('nuc', vc0=1, vc1=0.01)
            >>> print(m.tc)

            nuc_p_s0*w_none*(nuc_p_s0*vc1_nuc_none + 2*vc0_nuc_none)/2

            >>> m.freeze_parameters()
            >>> print(m.tc)

            nuc_p_s0*(0.005*nuc_p_s0 + 1.0)

        Here only the power output variable `nuc_p_s0` is left in the equation.
        All other symbols (all parameters) have been substituted with their
        respective numerical values.

        Excluding parameters through the `exceptions` argument causes a
        partial substitution:

        .. code-block:: python

            >>> m.freeze_parameters(exceptions=['vc0_nuc_none'])
            >>> print(m.tc)

            nuc_p_s0*(0.005*nuc_p_s0 + 1.0*vc0_nuc_none)

        '''

        for param in self.parameters():
            param._unfreeze_value()

        exceptions = [] if not exceptions else exceptions

        list_valid = self.parameters('name')
        list_invalid = set(exceptions) - set(list_valid)

        assert not list_invalid, ('Invalid names %s in exceptions parameter. '
                                  'Valid options are %s.'
                                  )%(', '.join(list_invalid),
                                     ', '.join(list_valid))

        param_list = [param for param, name in self.parameters(('', 'name'))
                      if not name in exceptions]

        for param in param_list:
            param._freeze_value()


    def _assert_slot_block_validity(self):
        '''
        If slot blocks are used, only the case with 2 blocks containing 2 slots
        each is implemented.
        '''

        # adding first non-slot/non-slot_block component
        slots_done = (len(self.comps) - hasattr(self, 'curt')
                      > len(self.slots) + len(self.slot_blocks))

        # check validity of time slot block definition
        if (self.slot_blocks and (
            len(self.slots) > 4 or (slots_done and len(self.slots) < 4))):

            raise RuntimeError('Number of time slots must be equal to 4 '
                               'if time slot blocks are used.')

        if len(self.comps) > 0:  # only when other components are added
            assert len(self.slot_blocks) in [0, 2], \
                        'Number of slot blocks must be 0 or 2.'


        if self.slot_blocks and slots_done:
            slots_per_block = Counter(s.block for s in self.slots.values())
            assert set(slots_per_block.values()) == set((2,)), \
                'Each slot block must be associated with exactly 2 slots.'


    @wrapt.decorator
    def _add_slots_to_kwargs(f, self, args, kwargs):

        if not 'slots' in kwargs:
            kwargs.update(dict(slots=self.slots))
        return f(*args, **kwargs)


    @property
    def df_comb(self):

        if self._df_comb is not None:
            return self._df_comb
        else:
            raise AttributeError('Model attribute df_comb not yet defined. '
                                 'Construct a valid model and call '
                                 'Model.generate_solve() or '
                                 'Model.init_constraint_combinations()')


    @df_comb.setter
    def df_comb(self, df_comb):
        if df_comb is None:
            return

        self._df_comb = df_comb.reset_index(drop=True)
        if self.nress:
            self.nress = len(self._df_comb)
        else:
            self.ncomb = len(self._df_comb)


    @_check_component_replacement
    def add_slot_block(self, name, repetitions):
        '''
        Add a time slot block to the model.

        %s
        '''

        self.slot_blocks.update({name: SlotBlock(name, repetitions)})


    @_update_component_list
    @_add_slots_to_kwargs
    @_check_component_replacement
    def add_storage(self, name, *args, **kwargs):
        r'''
        Add generic storage capacity to the model.

        %s

        '''  # Storage docstring added

        kwargs['_slot_blocks'] = self.slot_blocks
        self.storages.update({name: Storage(name, **kwargs)})


    @_update_component_list
    @_add_slots_to_kwargs
    @_check_component_replacement
    def add_plant(self, name, *args, **kwargs):
        r'''
        Add a dispatchable power plant to the model.

        %s
        '''  # Plant docstring added

        self.plants.update({name: Plant(name, **kwargs)})


    # note: no _update_component_list since slots alone make no model
    @_check_component_replacement
    def add_slot(self, name, *args, **kwargs):
        '''
        Add a time slot to the model.

        %s
        '''

        if 'block' in kwargs:
            bk = kwargs['block']
            assert bk in self.slot_blocks, 'Unknown block %s'%bk
            kwargs['block'] = self.slot_blocks[bk]

        elif self.slot_blocks:
            raise RuntimeError(('Error in `add_slot(%s)`: If any of the slots '
                                'are assigned to a block, all slots must be.'
                               )%name)

        if not 'weight' in kwargs:  # use default weight parameter
            kwargs['weight'] = self._slot_weights

        self.slots.update({name: Slot(name, **kwargs)})


    @_update_component_list
    def add_curtailment(self, slots):
        '''
        Add curtailment to the model. Must specify the time slots.

        This method is only used if curtailment is defined for a subset
        of time slots. Use the model parameter `curtailment=True` to enable
        curtailment globally.

        Parameters
        ----------
        slots : list
           list of time slot names, e.g. ``['day', 'night']``

        '''

        if self.curtailment:
            raise RuntimeError('Cannot manually add curtailment if model '
                               'level curtailment is True.')

        self._add_curtailment(slots)


    def _add_curtailment(self, slots):


        logger.debug('_add_curtailment with slots=%s'%str(slots))
        if not isinstance(slots, (dict)):
            slots = {slot: self.slots[slot] for slot in slots}

        self.curt.update({'curt': Curtailment('curt', slots=slots)})


    def _init_total_cost(self):
        '''
        Generate total cost and base lagrange attributes.

        Collects all cost components to calculate their total sum `tc`. Adds
        the equality constraints to the model's total cost to generate the base
        lagrange function `lagrange_0`.

        Costs and constraint expression of all components are re-initialized.
        This is important in case parameter values are frozen.
        '''

        comp_list = list(self.plants.values()) + list(self.storages.values())

        for comp in comp_list:
            comp._init_cost_component()
            comp._reinit_all_constraints()

        eq_cstrs = self.constraints.tolist('expr', is_equality_constraint=True)

        self.tc = sum(p.cc for p in comp_list)
        self.lagrange_0 = self.tc + sum(eq_cstrs)


# =============================================================================
# =============================================================================


    def _supply_cstr_expr_func(self, slot):
            '''
            Initialize the load constraints for a given time slot.
            Note: this accesses all plants, therefore method of the model class.
            '''

            total_chg = sum(store.pchg[slot]
                            for store in self.storages.values()
                            if slot in store.pchg)
            total_dch = sum(store.pdch[slot]
                            for store in self.storages.values()
                            if slot in store.pdch)

            equ = (slot.l.symb
                   + total_chg
                   - total_dch
                   - sum(plant.p[slot] for plant in self.plants.values()))

            if self.curt and slot in self.curt['curt'].p:
                equ += self.curt['curt'].p[slot]

            if hasattr(slot, 'vre'):
                equ -= slot.vre.symb * self.vre_scale.symb

            return slot.w.symb * equ


    def _init_supply_constraints(self):
        '''
        Defines a dictionary cstr_load {slot: supply constraint}
        '''

        for slot in self.slots.values():

            cstr = Constraint('supply', expr_func=self._supply_cstr_expr_func,
                              slot=slot, is_equality_constraint=True,
                              expr_args=(slot,), comp_name=slot.name)

            self.constraints.append(cstr)

# =============================================================================
# =============================================================================


    def generate_solve(self):
        '''
        Initialize the constraint combinations, generate the problems, and
        solve. This calls the following methods:

            - `Model.init_constraint_combinations()`
            - `Model.define_problems()`
            - `Model.solve_all()`
            - `Model.filter_invalid_solutions()`
            - `Model.generate_total_costs()`
            - `Model.cache.write(Model.df_comb)`

        '''


        if self.cache.file_exists:
            self.df_comb = self.cache.load()
        else:
            self.init_constraint_combinations(self.constraint_filt)
            self.define_problems()
            self.solve_all()
            self.filter_invalid_solutions()
            self.generate_total_costs()
            self.cache.write(self.df_comb)


    def _get_model_mutually_exclusive_cols(self):
        '''
        Expand model `_MUTUALLY_EXCLUSIVE` to plants and time slots.

        The initial list of constraint combinations is filtered only according
        to constraint combinations within each component separately
        (component `_MUTUALLY_EXCLUSIVE` dictionaries). Here,
        additional constraint combinations from different components are
        removed.

        Assuming only `'this'` as slottype.

        TODO: Integrate with constrcomb.CstrCombBase or derived class thereof.
        '''

        list_col_names = []

        dict_struct = {('comp_name', 'base_name'): {('slot',): ''}}
        cstrs_all = self.constraints.to_dict(dict_struct=dict_struct)

        for mename, me in self.mutually_exclusive.items():
            # expand to all components
            me_exp = [tuple((cstrs, name_cstr[0], me_slct[-1])
                       for name_cstr, cstrs in cstrs_all.items()
                       if name_cstr[1].endswith(me_slct[0]))
                      for me_slct in me]

            # all components of the combination's two constraints
            # for example, ('n', 'g'), 'curt' --> ('n', curt), ('g', curt)
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


    def init_constraint_combinations(self, constraint_filt=None):
        '''
        Generates dataframe `model.df_comb` with constraint combinations.

        1. Obtains relevant constraint combinations from components (see
           :func:`symenergy.core.component.Component._get_constraint_combinations`)
        2. Generates table corresponding to the full cross-product of all
           component constraint combinations.
        3. Filters constraint combinations according to the
           :attr:`model.mutually_exclusive` class attribute.

        This function initilizes the `symenergy.df_comb` attribute

        Parameters
        ----------
        constraint_filt : str
            :func:`pandas.DataFrame.query` string to filter the constraint
            activation columns of the `df_comb` dataframe. A list of relevant
            column names of a model object `m` can be retrieved through
            ``m.constraints('col', is_equality_constraint=False)``


        '''

        list_dfcomb = []
        for comp in self.comps.values():
            list_dfcomb.append(comp._get_constraint_combinations())

        list_dfcomb = [df for df in list_dfcomb if not df.empty]

        dfcomb = pd.DataFrame({'dummy': 1}, index=[0])

        for df in list_dfcomb:
            dfcomb = pd.merge(dfcomb, df.assign(dummy=1),
                              on='dummy', how='outer')

        logger.info('Length of merged df_comb: %d'%len(dfcomb))

        # filter according to model mutually_exclusive
        logger.info('*'*30 + 'model filtering' + '*'*30)
        model_mut_excl_cols = self._get_model_mutually_exclusive_cols()
        dfcomb = filter_constraint_combinations(dfcomb, model_mut_excl_cols)

        self.df_comb = dfcomb.drop('dummy', axis=1)

        if constraint_filt:
            self.df_comb = self.df_comb.query(constraint_filt)

        self.ncomb = len(self.df_comb)

        logger.info('Remaining df_comb rows: %d' % self.ncomb)

#
#    def get_variabs_params(self):
#        '''
#        Generate lists of parameters and variables.
#
#        Gathers all parameters and variables from its components.
#        This is needed for the definition of the linear equation system.
#        '''
#
##        self.params = {par: comp
##                       for comp in self.comps.values()
##                       for par in comp.get_params_dict()}
#
#        # add time-dependent variables
##        self.variabs = {var: comp
##                        for comp in self.comps.values()
##                        for var in comp.get_variabs()}
#
#        # parameter multips
##        self.multips = {cstr.mlt: comp for cstr, comp in self.constrs.items()}
#        # supply multips
##        self.multips.update({cstr.mlt: slot
##                             for slot, cstr in self.cstr_supply.items()})



# =============================================================================
#     Various solver-related methods
# =============================================================================

    def solve(self, x):

        # substitute variables with binding positivitiy constraints
        cpos = self.constraints.tolist(('col', ''),
                                       is_positivity_constraint=True)
        subs_zero = {cstr.expr_0: sp.Integer(0) for col, cstr
                     in cpos if x[cstr.col]}

        mat = derive_by_array(x.lagrange, x.variabs_multips)
        mat = sp.Matrix(mat).expand()
        mat = mat.subs(subs_zero)

        variabs_multips_slct = list(OrderedSet(x.variabs_multips) - OrderedSet(subs_zero))

        A, b = linear_eq_to_matrix(mat, variabs_multips_slct)

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
            x = self.df_comb.iloc[0]

        if not self.nworkers:
            self.df_comb['result'] = self.call_solve_df(self.df_comb)
        else:
            func = self.wrapper_call_solve_df
            self.df_comb['result'] = parallelize_df(self.df_comb, func,
                                                    nworkers=self.nworkers)

# =============================================================================
# =============================================================================


    def _subs_total_cost(self, x):
        '''
        Substitutes solution into TC variables.
        This expresses the total cost as a function of the parameters.
        '''

        res, var = x.result, x.variabs_multips

        dict_var = {var: res[ivar]
                    if not isinstance(res, sp.sets.EmptySet)
                    else np.nan for ivar, var
                    in enumerate(var)}

        MP_COUNTER.increment()

        return self.tc.copy().subs(dict_var)


    def call_subs_tc(self, df):

        return df.apply(self._subs_total_cost, axis=1)


    def wrapper_call_subs_tc(self, df, *args):

        name = 'Substituting total cost'
        ntot = self.nress
        return log_time_progress(self.call_subs_tc)(self, df, name, ntot)


    def generate_total_costs(self):
        '''
        Substitute result variable expressions into total costs.

        This adds an additional total cost column ``'tc'`` to the
        :attr:`symenergy.core.model.Model.df_comb` table. The total cost is
        calculated by substituting the solutions for all variables into
        the total cost expression ``Model.tc`` (for each constraint combination).

        The execution is parallelized. The ``Model.nworkers`` attribute defines
        the number of workers for multiprocessing.
        '''

        logger.info('Generating total cost expressions...')

        df = self.df_comb[['result', 'variabs_multips', 'idx']]

        if not self.nworkers:
            self.df_comb['tc'] = self.call_subs_tc(df)
        else:
            func = self.wrapper_call_subs_tc
            self.df_comb['tc'] = parallelize_df(df, func,
                                                nworkers=self.nworkers)


# =============================================================================
# =============================================================================


    def construct_lagrange(self, row):

        lagrange = self.lagrange_0
        active_cstrs = row[row == 1].index.values
        lagrange += sum(expr for col, expr
                        in self.constraints.tolist(('col', 'expr'))
                        if col in active_cstrs)

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
        return log_time_progress(self.call_construct_lagrange
                                 )(self, df, name, ntot)


# =============================================================================
# =============================================================================

    def _get_variabs_multips_slct(self, lagrange):
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
        MP_COUNTER.increment()
        list_vm = [ss for ss in lfs
                   if ss in self.variables.tolist('symb')
                          + self.constraints.tolist('mlt')]
        return sorted(list_vm, key=str)


    def _call_get_variabs_multips_slct(self, df):

#        res = list(map(self.get_variabs_multips_slct, df))
        return df.apply(self._get_variabs_multips_slct)


    def _wrapper_call_get_variabs_multips_slct(self, df, *args):

        name = 'Get variabs/multipliers'
        ntot = self.ncomb
        func = self._call_get_variabs_multips_slct
        return log_time_progress(func)(self, df, name, ntot)

# =============================================================================
# =============================================================================



    def fix_linear_dependencies(self, x):
        '''
        All solutions showing linear dependencies are set to zero. See doc
        of symenergy.core.model.Model.get_mask_linear_dependencies
        '''

        MP_COUNTER.increment()

        if __name__ == '__main__':
            x = self.df_comb.iloc[0]

        if x.code_lindep == 0:
            list_res_new = x.result

        elif x.code_lindep == 3:
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
                    for res in list_res_new:
                        res.subs(dict.fromkeys(free_symbs, sp.numbers.Zero()))

                    collect[list_var[nres]] = ', '.join(map(str, free_symbs))

            if collect:
                logger.debug('idx=%d'%x.idx)
                for res, var in collect.items():
                    logger.debug(('     Solution for %s contained variabs '
                                  '%s.')%(res, var))
        else:
            raise ValueError('code_lindep must be 0, 3, or 1')

        return list_res_new


    def call_fix_linear_dependencies(self, df):

        return df.apply(self.fix_linear_dependencies, axis=1)


    def wrapper_call_fix_linear_dependencies(self, df, *args):

        name = 'Fix linear dependencies'
        ntot = self.nress
        func = self.call_fix_linear_dependencies
        return log_time_progress(func)(self, df, name, ntot)





# =============================================================================
# =============================================================================


    def define_problems(self):
        '''
        For each combination of constraints, define

        * the Lagrange functions (new column *lagrange*
          in the ``df_comb`` table)
        * the endogenous (dependent) variables and multipliers
          (new column *variabs_multips* in the ``df_comb`` table)

        '''

        logger.info('Defining lagrangians...')
        df = self.df_comb[self.constrs_cols_neq]
        if not self.nworkers:
            self.df_comb['lagrange'] = self.call_construct_lagrange(df)
        else:
            func = self.wrapper_call_construct_lagrange
            nworkers = self.nworkers
            self.df_comb['lagrange'] = parallelize_df(df, func,
                                                      nworkers=nworkers)

        logger.info('Getting selected variables/multipliers...')
        df = self.df_comb.lagrange
        if not self.nworkers:
            self.list_variabs_multips = self.call_get_variabs_multips_slct(df)
            self.df_comb['variabs_multips'] = self.list_variabs_multips
        else:
            func = self._wrapper_call_get_variabs_multips_slct
            nworkers = self.nworkers
            self.df_comb['variabs_multips'] = parallelize_df(df, func,
                                                             nworkers=nworkers)

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


    def get_mask_linear_dependencies(self):
        '''
        Solutions of problems containing linear dependencies.

        In case of linear dependencies SymPy returns solutions containing
        variables which we are actually solving for. To fix this, we
        differentiate between two cases:

        0. No dependencies
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
        3. Different components but same component classes. If multiple idling
           storage plants are present, their mulitpliers show linear
           dependencies.

        Returns
        -------
        code_lindep : pandas.Series
           Series with same length as ``df_comb`` with linear dependency codes
           as defined above
        '''

        res_vars = self.df_comb[['result', 'variabs_multips', 'idx']].copy()

        # map variable/multiplier -> component
        dict_vm_cp = {**self.variables.to_dict({('symb',): 'comp_name'}),
                      **self.constraints.to_dict({('mlt', ): 'comp_name'})}
        dict_vm_cp = {vm: self.comps[cp] for vm, cp in dict_vm_cp.items()}

        # map variable/multipler -> component class
        dict_vm_cl = {vm: cp.__class__ for vm, cp in dict_vm_cp.items()}

        return_series = lambda *args: pd.Series(args,
                                               index=['ncompunq', 'nclassunq'])

        # for each individual solution, get residual variables/multipliers
        def get_residual_vars(x):
            if __name__ == '__main__':
                x = res_vars.iloc[0]

            varmlt = x.variabs_multips
            result = x.result

            # identify free variables/multipliers in results
            resvars = [r.free_symbols & set(varmlt) for r in result]
            # add the corresponding solution variable to all non-empty sets
            resvars = [rv | {vm} for rv, vm in zip(resvars, varmlt) if rv]
            if not resvars:
                return return_series(0, 0)
            # get components corresponding to symbols (unique for each result)
            rescomps = [set(map(lambda x: dict_vm_cp[x], rv)) for rv in resvars]
            # maximum number of distinct components
            ncompunq = max(map(len, rescomps))
            # get classes corresponding to symbols (unique for each result)
            resclass = [set(map(lambda x: dict_vm_cl[x], rv)) for rv in resvars]
            # maximum number of distinct classes
            nclassunq = max(map(len, resclass))

            return return_series(ncompunq, nclassunq)

        max_cnt = res_vars.apply(get_residual_vars, axis=1)

        # generate lindep codes
        max_cnt['code_lindep'] = 0
        mask_1 = (max_cnt.ncompunq < 2) & (max_cnt.ncompunq > 0)
        max_cnt.loc[mask_1, 'code_lindep'] = 1
        mask_2 = (max_cnt.ncompunq >= 2) & (max_cnt.nclassunq >= 2)
        max_cnt.loc[mask_2, 'code_lindep'] = 2
        mask_3 = (max_cnt.ncompunq >= 2) & (max_cnt.nclassunq <= 1)
        max_cnt.loc[mask_3, 'code_lindep'] = 3

        return max_cnt.code_lindep


    def filter_invalid_solutions(self):
        '''
        Analyzes the model result expressions to filter invalid rows.

        This method modifies and shortens the ``Model.df_comb`` table.

        * Identify empty solutions as returned by the linsolve method
        * Remove empty solutions from ``Model.df_comb``. Invalid solutions
          are kept in the ``Model.df_comb_invalid`` dataframe.
        * Analyze and classify remaining solutions with respect to linear
          dependencies of solutions
          (:func:`symenergy.core.model.Model.get_mask_linear_dependencies`).
        * Fix results with fixable linear dependencies.
        '''

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
        nkey1, nkey2, nkey3 = ((mask_lindep == 1).sum(),
                               (mask_lindep == 2).sum(),
                               (mask_lindep == 3).sum())
        logger.warning(('Number of solutions with linear dependencies: '
          'Key 1: {:d} ({:.1f}%), Key 2: {:d} ({:.1f}%), Key 3: {:d} ({:.1f}%)'
                       ).format(nkey1, nkey1/ncomb0*100,
                                nkey2, nkey2/ncomb0*100,
                                nkey3, nkey3/ncomb0*100))

        self.df_comb['code_lindep'] = mask_lindep
        self.df_comb = self.df_comb.loc[-(self.df_comb.code_lindep == 2)]

        self.nress = len(self.df_comb)

        # adjust results for single-component linear dependencies
        if not self.nworkers:
            self.df_comb['result'] = \
                    self.call_fix_linear_dependencies(self.df_comb)
        else:
            func = self.wrapper_call_fix_linear_dependencies
            nworkers = self.nworkers
            self.df_comb['result'] = parallelize_df(self.df_comb, func,
                                                    nworkers=nworkers)


    def print_results(self, idx, df=None, slct_var_mlt=None, substitute=None):
        '''
        Print result expressions for all variables and multipliers for a
        certain constraint combination index.

        Parameters
        ----------
            idx : int
                index of the constraint combination for which the results are to
                be printed
            df : df
                DataFrame containing the results and the index; defaults to
                the model's `df_comb` table
            substitute : dict
                substitutions to be performed prior to expression simplification
                and printing; main use case: setting the energy cost parameter
                ``ec`` of the storage class to zero.

        The input DataFrame must have the following columns:

        * `variabs_multips`: iterable of variable and multiplier symbols
          for which the results are to be printed
        * `result`: list of expressions corresponding for each of the
          `variabs_multips` symbols
        '''

        if not isinstance(df, pd.DataFrame):
            df = self.df_comb

        x = df.reset_index().query('idx == %d'%idx).iloc[0]

        if not substitute:
            substitute = {}

        resdict = dict(zip(map(str, x.variabs_multips), x.result))
        if slct_var_mlt:
            resdict = {var: res for var, res in resdict.items()
                       if var in slct_var_mlt}

        resdict = pd.DataFrame.from_dict(resdict, orient='index'
                                         ).reset_index().sort_values('index').set_index('index').to_dict()[0]

        for var, res in resdict.items():

            print('*'*20, var, '*'*20)
            print(sp.simplify(res.subs(substitute)))


    def __repr__(self):

        ret = str(self.__class__)
        return ret

#    def get_all_is_capacity_constrained(self):
#        ''' Collects capacity constrained variables of all components. '''
#
#        return [var
#                for comp in self.comps.values()
#                for var in comp.get_is_capacity_constrained()]

#    def get_all_is_positive(self):
#        ''' Collects positive variables of all components. '''
#
#        is_positive_comps = [var
#                             for comp in self.comps.values()
#                             for var in comp.get_is_positive()]
#
#        return is_positive_comps


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


    @hexdigest
    def get_model_hash_name(self):

        hash_input = ''.join(comp._get_hash_name()
                             for comp in self.comps.values())
        hash_input += str(self.lagrange_0)
        hash_input += self.constraint_filt

        return hash_input


# add component class docs to the component adder docstrings
for addermethod, compclass in [(Model.add_storage, Storage),
                               (Model.add_plant, Plant),
                               (Model.add_slot, Slot),
                               (Model.add_slot_block, SlotBlock)]:

    doc = addermethod.__doc__
    classdoc = textwrap.dedent(compclass.__doc__)
    lines = doc.split('\n')
    ind = min(len(line) - len(line.strip(' ')) for line in lines
              if not line == '')
    classdoc = textwrap.indent(classdoc, ' ' * ind)

    addermethod.__doc__ = doc % classdoc


