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

# Definition of power plants

class Component():
    '''
    Make sure that all children implement PARAMS, VARIABS AND MULTIPS
    '''
    
    
    def get_params(self):
        
        return [getattr(self, par) for par in self.PARAMS
                if par in self.__dict__.keys()]

    def get_variabs(self):
        
        return [getattr(self, var) for var in self.VARIABS
                if var in self.__dict__.keys()]
        
    def get_multipls(self):
        
        return [getattr(self, mlt) for mlt in self.MULTIPS
                if mlt in self.__dict__.keys()]


class Plant(Component):
    
    '''
    Does know about slots.
    '''
    '''
    All plants have:
        * symbol power production p
        * symbol costs vc_0 and vc_1
        * cost component (vc_0 + 0.5 * vc_1 * p) * p
        * multiplier > 0
    Some plants have:
        * symbol capacity C
        * value capacity
        * multiplier power p <= capacity C
    '''
    PARAMS = ['vc0', 'vc1', 'C']
    VARIABS = ['p']
    MULTIPS = ['lambda_C', 'lambda_pos']
    
    def __init__(self, name, vc0, vc1, capacity=False):

        self._name = name
        
        self.has_capacity = False        
        
        self.param_values = {'vc0': vc0, 'vc1': vc1}

        self.init_symbol_power()
        self.init_symbols_costs()
        self.init_shadow_price_positive_power()

        self.init_positive_power_constraint()


        if capacity:
            self.param_values.update({'C': capacity})
            
            self.has_capacity=True            
            self.init_symbol_capacity()
            self.init_shadow_price_maximum_power()
            
            self.init_maximum_power_constraint()
            
        self.init_is_capacity_constrained()
        self.init_is_positive()            
            
    def init_is_capacity_constrained(self):
        
        self.is_capacity_constrained = (self.p.name
                                        if 'C' in self.param_values.keys()
                                        else None)
            
    def init_is_positive(self):        

        self.is_positive = self.p.name
        
            
    def init_symbol_capacity(self):
        
        setattr(self, 'C', sp.symbols('%s_C'%self._name, positive=True))        

    def init_symbol_power(self):
        
        setattr(self, 'p', sp.symbols('%s_p'%self._name, positive=True))

    def init_symbols_costs(self):
        '''
        Set constant and linear components of variable costs.
        '''

        self.vc0 = sp.symbols('%s_vc0'%self._name, positive=True)
        self.vc1 = sp.symbols('%s_vc1'%self._name, positive=True)

        self.vc = self.vc0 + self.vc1 * self.p

        self.cc = sp.integrate(self.vc, self.p)

    def init_positive_power_constraint(self):
        
        self.cstr_pos = self.lambda_pos * self.p

    def init_maximum_power_constraint(self):

        self.cstr_C = self.lambda_C * (self.p - self.C)

    def init_shadow_price_positive_power(self):
        
        self.lambda_pos = sp.symbols('%s_lambda_pos'%self._name)

    def init_shadow_price_maximum_power(self):
        
        self.lambda_C = sp.symbols('%s_lambda_C'%self._name)

    def _subs_cost(self, symb, p=False):
        
        dict_subs = {getattr(self, vc): self.param_values[vc]
                     for vc in ['vc0', 'vc1']}
        
        if p:
            dict_subs.update({getattr(self, '%s_p'%self._name): p})
        
        return symb.subs(dict_subs)


    def __repr__(self):
        
        ret = ('Plant: %s'%self._name + '\n'
               'Symbol: %s'%str(getattr(self, 'p')) + '\n'
               'Variable cost: %s = %s'%(str(self.vc), str(self._subs_cost(self.vc))) + '\n'
               'Total cost component: %s = %s'%(str(self.cc), str(self._subs_cost(self.cc))) + '\n'
               'Shadow price positive power: %s'%str(self.lambda_pos)
               )
        if self.has_capacity:
            
            ret += ('\nCapacity: %s=%s'%(self.C, self.param_values['C']) + '\n')
            
        return ret

class Slot(Component):
    '''
    Doesn't know about plants.
    '''

    PARAMS = ['vre', 'l']
    VARIABS = []
    MULTIPS = ['pi']

    def __init__(self, name, load, vre):
        
        self.name = str(name)
        
        self.init_load()
        self.init_vre()
        
        self.init_shadow_price_load()
 
        self.param_values = {'l': load, 'vre': vre}

    def init_shadow_price_load(self):
        
        self.pi = sp.symbols('pi')

    def init_vre(self):
        
        self.vre = sp.symbols('%s_vre'%self.name, constant=True)

    def init_load(self):
        
        self.l = sp.symbols('%s_l'%self.name, constant=True)
        
    def __repr__(self):
        
        ret = ('Slot: %s'%self.name + '\n'
               'Load: %s=%s'%(self.l, self.param_values['l']) + '\n'
               'VRE: %s=%s'%(self.vre, self.param_values['vre']) + '\n'
               'Shadow price load: %s'%self.pi)
        return ret
        
        
class Model:
    
    def __init__(self):

        # temporal structure stuff

        self.plants = {}
        self.slots = {}

        self.comps = []

    def update_component_list(f):
        def wrapper(self, *args, **kwargs):
            f(self, *args, **kwargs)
#            self.comps = self.plants.update(self.slots)
            
            self.comps = self.plants.copy()
            self.comps.update(self.slots)
            print('self.comps', self.comps, self.plants, self.slots)
            print('CALLING get_variabs_params')
            self.get_variabs_params()
            self.init_total_param_values()
            
            
        return wrapper
    

    @update_component_list
    def add_plant(self, name, *args, **kwargs):
        
        self.plants.update({name: Plant(name, **kwargs)})
        self.init_total_cost()
        

    @update_component_list
    def add_slot(self, name, *args, **kwargs):

        self.slots.update({name: Slot(name, **kwargs)})
        
    def init_total_param_values(self):
        
        param_values_plants = {getattr(p, '%s'%(param)): param_value
                               for pname, p in self.plants.items()
                               for param, param_value
                               in p.param_values.items()}
        
        param_values_slots = {getattr(p, '%s'%(param)): param_value
                              for pname, p in self.slots.items()
                              for param, param_value
                              in p.param_values.items()}
        
        self.param_values = {}
        self.param_values.update(param_values_plants)
        self.param_values.update(param_values_slots)

    def init_total_cost(self):
        
        self.tc = sum(p.cc for p in self.plants.values())         
            
    def init_load_constraints(self):
        '''
        Initialize the load constraints for each time slot.
        
        For all time slots this loops over all (relevant) plants and
        combines the power symbols.
        '''
        
        self.cstr_load = self.slots[0].l - self.slots[0].vre
        
        self.cstr_load -= sum(plant.p for plant in self.plants.values())
        
        self.cstr_load *= self.slots[0].pi

    def get_constraint_combinations(self):
        
        # greater zero constraints
        self.constrs = [p.lambda_pos for p in self.plants.values() if 'lambda_pos' in p.__dict__]
        # smaller capacity constraints
        self.constrs += [p.lambda_C for p in self.plants.values() if 'lambda_C' in p.__dict__]

        self.constrs = ['active_%s'%cc.name for cc in self.constrs]

        self.df_comb = pd.DataFrame(list(itertools.product(*[[0, 1]
                                    for cc in self.constrs])),
                                    columns=self.constrs)
    
    def get_variabs_params(self):
        '''
        Generate lists of parameters and variables separately.
        This is needed
        for the definition of the linear equation system.
        '''
        
        self.params = [par
                       for component in self.comps.values()
                       for par in component.get_params()
                       if self.comps]

        self.variabs = [var
                        for component in self.comps.values()
                        for var in component.get_variabs()]
         
        self.multips = [mtp
                        for component in self.comps.values()
                        for mtp in component.get_multipls()]
        
        
    def construct_lagrange(self, row):
        
        
        slct_constr = row[1].to_dict()
        
        lagrange = self.tc + self.cstr_load
        
        
        for constr, is_active in slct_constr.items():
            
            if is_active:
                
                # get relevant constraint...
                # plant and constraint are identified through constraint
                # symbol name...
                # TODO: Not sure about this...
                
                
                _, pname, _, cname = constr.split('_')
                
                cstr = getattr(self.plants[pname], 'cstr_%s'%cname)
#                    print('%s is active, adding.'%constr)
                lagrange += cstr
    
        return lagrange


    def get_variabs_multips_slct(self, lagrange):
    
        lfs = lagrange.free_symbols
        variabs_slct = [ss for ss in lfs if ss in self.variabs]
        multips_slct = [ss for ss in lfs if ss in self.multips]
#        params_slct = [ss for ss in lfs if ss in self.params]

        return variabs_slct, multips_slct


    def solve(self, lagrange, variabs_slct, multips_slct):

    
        mat = sp.tensor.array.derive_by_array(lagrange, variabs_slct + multips_slct)
        mat = sp.Matrix(mat)
        mat = mat.expand()
        
        A, b = sp.linear_eq_to_matrix(mat, variabs_slct + multips_slct)
            
        return sp.linsolve((A, b), variabs_slct + multips_slct)

    def loop_constraint_combinations(self):
        '''
        For each combination of constraints, get the linear equation system.
        '''

        row = list(self.df_comb.iterrows())[0]
        for row in self.df_comb.iterrows():
            
            self.row = row
            row = self.row
            
            lagrange = self.construct_lagrange(row)
            
            variabs_slct, multips_slct = self.get_variabs_multips_slct(lagrange)
            
            row_result = self.solve(lagrange, variabs_slct, multips_slct)

            print('Result this constraint combination: ', row_result)

            self.df_comb.loc[row[0], 'result'] = [row_result.copy()]
                    
            self.df_comb.loc[row[0], 'lagrange'] = lagrange

            self.df_comb.loc[row[0], 'variabs_multips'] = [[variabs_slct.copy()
                                                            + multips_slct.copy()]]

            

            self.df_comb.loc[row[0], 'tc'] = self.subs_total_cost(row_result, variabs_slct, multips_slct)

    def filter_invalid_solutions(self):

        check_emptyset = lambda res: isinstance(res, sp.sets.EmptySet)

        mask_invalid = self.df_comb.result.apply(check_emptyset)   
            
        df_comb_invalid = self.df_comb.loc[mask_invalid]
        
        print('The following constraint combinations are invalid:\n',
              df_comb_invalid[self.constrs])
                                                  
        self.df_comb = self.df_comb.loc[-mask_invalid]

    def subs_total_cost(self, result, variabs_slct, multips_slct):
        '''
        Substitutes solution into TC variables.
        This expresses the total cost as a function of the parameters.
        '''
        
        dict_var = {var: list(result)[0][ivar]
                    if not isinstance(result, sp.sets.EmptySet)
                    else np.nan for ivar, var
                    in enumerate(variabs_slct + multips_slct)}

        return self.tc.copy().subs(dict_var)


    def __repr__(self):
        
        ret = ('Total cost: tc=%s'%self.tc + '\n'
               'Load constraints: %s'%self.cstr_load
               )
        return ret

    def get_list_is_capacity_constrained(self):
        return [pp.is_capacity_constrained
                for _, pp in self.plants.items()
                if pp.is_capacity_constrained]

    def get_list_is_positive(self):
        return [pp.is_positive
                for _, pp in self.plants.items()
                if pp.is_positive]

class Evaluator():
    
    def __init__(self, model, select_x, select_y, x_vals):
        
        self.model = model
        self.select_x = select_x
        self.select_y = select_y
        self.x_vals = x_vals
        
        
        self.dfev = model.df_comb.copy()
        
        # get all parameters to be set to a constant value
        self.params_set = [pp for pp in self.model.params
                           if not pp == select_x]
        
        # get param values corresponding to params_set in the right order
        self.params_values_set = [self.model.param_values[par]
                                  for par in self.params_set]
        
        
    def get_evaluated_lambdas(self):
        '''
        For each dependent variable and total cost get a lambda function
        evaluated by constant parameters. This subsequently evaluated 
        for all x_pos.
        
        Generated attributes:
            - df_lam_plot: Holds all lambda functions for each dependent variable and each constraint combination.
        '''
            
        # generate functions only dependent on the indepent variable
        for slct_eq_0 in ['tc'] + list(map(str, self.model.variabs + self.model.multips)):
            
            print('Extracting solution for %s'%slct_eq_0)
            
            slct_eq = slct_eq_0.name if not isinstance(slct_eq_0, str) else slct_eq_0
        
            # function idx depends on constraint, since not all constraints
            # contain the same functions
            if slct_eq != 'tc':

                self.dfev[slct_eq] = self.dfev.apply(lambda x: self._get_func_from_idx(x, slct_eq), axis=1)

            self.dfev['%s_expr_plot'%slct_eq] = self.dfev[slct_eq].apply(self._subs_param_values)

            self.dfev[slct_eq + '_lam_plot'] = self.dfev[slct_eq + '_expr_plot'].apply(lambda res_plot: sp.lambdify(self.select_x, res_plot, modules=['numpy']))

        df_lam_plot = self.dfev.set_index(list(map(str, self.model.constrs))).copy()[[c for c in self.dfev.columns if isinstance(c, str) and '_lam_plot' in c]]

        df_lam_plot = df_lam_plot.stack().reset_index().rename(columns={'level_%d'%len(self.model.constrs): 'func', 0: 'lambd'})        
        df_lam_plot = df_lam_plot.set_index(self.model.constrs + ['func'])
            
        self.df_lam_plot = df_lam_plot    
            
    def _get_func_from_idx(self, x, slct_eq):
        '''
        Get result expression corresponding to the selected variable slct_eq.
        
        From the result set of row x, get the expression corresponding to the
        selected variable/multiplier slct_eq. This first finds the index
        of the corresponding expression through comparison with slct_eq and
        then returns the expression itself.
        '''

        mv_list_str = [var.name for var in x.variabs_multips[0]]
        
        if (slct_eq in mv_list_str and
            not isinstance(x.result, sp.sets.EmptySet)):
            
            idx = mv_list_str.index(slct_eq)                
            
            func = list(x.result)[0][idx]
            
            return func
        else:
            return np.nan

    def _subs_param_values(self, x):
        '''
        Substitutes all parameter values
        except for the one selected as
        single independent variable.
        '''
        
        print(x)
        if isinstance(x, float) and np.isnan(x):
            return np.nan
        else:
            return x.subs({kk: vv for kk, vv
                           in self.model.param_values.items()
                           if kk in self.params_set})

    def expand_rows(self, lam_plot):
    
        print(lam_plot)
        y_vals = lam_plot.iloc[0](self.x_vals)
        
        if isinstance(y_vals, float):
            y_vals = np.ones(self.x_vals.shape) * y_vals
            
        return pd.Series(y_vals, index=pd.Index(self.x_vals))
    
    
    def expand_data_to_x_vals(self):
        
        # expand all data to selected values
        group_levels = self.model.constrs + ['func']
        dfg = self.df_lam_plot.groupby(level=group_levels)['lambd']
        df_exp_0 = dfg.apply(self.expand_rows).reset_index()
        
        df_exp_0 = df_exp_0.rename(columns={'level_%d'%(len(self.model.constrs) + 1): select_x.name})

        self.df_exp = df_exp_0
        

    def _get_constraints_active(self):
        '''
        Create binary columns depending on whether the plant constraints
        are active or not.
        '''
        
        self.is_capacity_constrained = self.model.get_list_is_capacity_constrained()
        self.is_positive = self.model.get_list_is_positive()
        
        get_var = lambda x: x.split('_lam_')[0]
        set_constr = lambda x, lst: 1 if x in getattr(self, lst) else 0
        
        for lst in ['is_positive', 'is_capacity_constrained']:
                    
            constr_act = (self.df_exp.func.apply(get_var)
                                          .apply(lambda x: set_constr(x, lst)))
            self.df_exp[lst] = constr_act


    def _get_mask_valid_solutions(self):

        mask_valid = pd.Series(1, index=self.df_exp.index)

        self.df_exp['mv_0'] = mask_valid

        # filter positive
        msk_pos = self.df_exp.is_positive == 1
        constraint_met = self.df_exp.loc[msk_pos].lambd >= 0
        mask_valid.loc[msk_pos] *= constraint_met

        self.df_exp['mv_pos'] = mask_valid

        # filter greater n
        # TODO: get capacity from component objects; need to add *all* capacities of the model as columns!!!!
        dict_cap = {pp: pp.C for _, pp in m.plants.items()
                        if 'C' in pp.__dict__}
        msk_cap_cstr = self.df_exp.is_capacity_constrained == 1
        for pp, C in dict_cap.items():
            # things are different depending on whether or not select_x is the corresponding capacity
            if select_x is C:
                constraint_met = (self.df_exp.loc[msk_cap_cstr].lambd
                                  <= self.df_exp.loc[msk_cap_cstr,
                                                     select_x.name])
            else:
                val_cap = self.model.param_values[C]
                constraint_met = (self.df_exp.loc[msk_cap_cstr].lambd
                                  <= val_cap)
            
            mask_valid.loc[msk_cap_cstr] *= constraint_met

        self.df_exp['mv_n'] = mask_valid
        
        self.df_exp['mask_valid'] = mask_valid

        # consolidate mask by constraint combination and x values
        index = self.model.constrs + [select_x.name]
        mask_valid = self.df_exp.pivot_table(index=index, values='mask_valid',
                                             aggfunc=min)
        
        self.df_exp.drop('mask_valid', axis=1, inplace=True)

        return mask_valid

    def enforce_constraints(self):
        '''
        Discard solutions which violate any of the
            * positive
            * capacity 
        constraints.
        TODO: Ideally this would be modular and part of the components.
        '''
        self._get_constraints_active()

        mask_valid = self._get_mask_valid_solutions()

        self.df_exp = self.df_exp.join(mask_valid, on=mask_valid.index.names)
        
        self.df_exp.loc[self.df_exp.mask_valid == 0, 'lambd'] = np.nan
        
    def combine_constraint_names(self):
        
        constr_name = pd.DataFrame(index=self.df_exp.index)
        for const in self.model.constrs:
        
            constr_name[const] = const + '=' + self.df_exp[const].astype(str)

        join = lambda x: ', '.join(x)
        self.df_exp['const_comb'] = constr_name.apply(join, axis=1)

    def get_cost_optimum(self):
        
        tc = self.df_exp.loc[self.df_exp.func == 'tc_lam_plot'].copy()
        
        tc_min = tc.groupby('b_C', as_index=0).apply(lambda x: x.nsmallest(1, 'lambd'))
        
        df_exp_min = pd.merge(tc_min[self.model.constrs + ['b_C']],
                              self.df_exp,
                              on=['b_C'] + self.model.constrs, how='inner')
        
        df_exp_min['const_comb'] = 'cost_optimum'
        
        self.df_exp = pd.concat([self.df_exp, df_exp_min], axis=0, sort=False)


m = Model()

self = m

m.add_plant(name='n', vc0=0.2, vc1=0.01, capacity=3)
m.add_plant(name='g', vc0=0.5, vc1=0.04)

m.add_slot(name=0, load=5, vre=0)
m.init_load_constraints()
m.get_constraint_combinations()

m.loop_constraint_combinations()

m.filter_invalid_solutions()

# %%

select_y = m.plants['g'].p
select_x = m.plants['b'].C
x_vals = np.linspace(0, 6, 101)
ev = Evaluator(m, select_x, select_y, x_vals)

self = ev

ev.get_evaluated_lambdas()

ev.df_lam_plot

ev.expand_data_to_x_vals()

ev.enforce_constraints()

ev.combine_constraint_names()

ev.get_cost_optimum()

# %% DEFINE COMBINATION OF CONSTRAINTS

import pyAndy.core.plotpage as pltpg

data_kw = dict(ind_axx=[select_x.name], ind_pltx=['func'], ind_plty=[],
               series=['const_comb'], values=['lambd'],
               aggfunc=np.mean, harmonize=True)
page_kw = dict(left=0.05, right=0.99, bottom=0.1, top=0.8)
plot_kw = dict(kind_def='LinePlot', stacked=False, on_values=True,
               marker='.', xlabel=select_x.name)


do = pltpg.PlotPageData.from_df(df=ev.df_exp, **data_kw)
plt0 = pltpg.PlotTiled(do, **plot_kw, **page_kw)

do_tc = do.copy()
do_tc.data = do_tc.data.loc[do_tc.data.index.get_level_values('func') == 'tc_lam_plot']
plttc = pltpg.PlotTiled(do_tc, **plot_kw, **page_kw)


for ix, namex, iy, namey, plot, ax, kind in plt0.get_plot_ax_list() + plttc.get_plot_ax_list():
    ''''''

    name_lin = ('lambd', 'cost_optimum')

    if name_lin in plot.linedict.keys():
    
        lin = plot.linedict[name_lin]
    
        lin.set_marker('o')
        lin.set_linewidth(0)




