#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains the symenergy component class.


Part of symenergy. Copyright 2018 authors listed in AUTHORS.
"""



class Component():
    '''
    Make sure that all children implement PARAMS, VARIABS AND MULTIPS
    '''
    def get_params_dict(self, attr=tuple()):

        attr = tuple(attr) if isinstance(attr, str) else attr

        param_objs = \
                [getattr(self, param_name)
                 for param_name in self.PARAMS_TIME + self.PARAMS
                 if hasattr(self, param_name)]

        if len(attr) == 1:
            return [getattr(par, attr[0])
                    for par in param_objs]
        elif len(attr) == 2:
            return {getattr(par, attr[0]): getattr(par, attr[1])
                    for par in param_objs}
        else:
            return param_objs

    def get_is_capacity_constrained(self):
        '''
        Returns a tuple of all variables defined by the MAP_CAPACITY dict.

        Only include if the capacity is defined.
        '''

        return tuple(var
                     for cap_name, var_names in self.MAP_CAPACITY.items()
                     for var_name in var_names
                     if hasattr(self, var_name) and hasattr(self, cap_name)
                     for slot, var in getattr(self, var_name).items())

    def get_is_positive(self):
        '''
        Returns a tuple of all variables defined by the VARIABS_POSITIVE list.
        '''

        return tuple(var
                     for var_name in self.VARIABS_POSITIVE
                     if hasattr(self, var_name)
                     for slot, var in getattr(self, var_name).items())

    def get_mutually_exclusive_cstrs(self):
        '''
        This expands the pairs from the MUTUALLY_EXCLUSIVE class attribute
        to all constraint columns.
        '''

        dict_cstrs = {key: attr for key, attr in self.__dict__.items() # TODO: this is no good !!
                      if key.startswith('cstr_')}

        mutually_exclusive = [(cstr1, cstr2) for cstr1, cstr2 in self.MUTUALLY_EXCLUSIVE
                              if 'cstr_%s'%cstr1 in dict_cstrs
                              and 'cstr_%s'%cstr2 in dict_cstrs]

        mutually_exclusive_cols = []
        for cstr1, cstr2 in mutually_exclusive:

            for slot, cstr1_obj in dict_cstrs['cstr_%s'%cstr1].items():

                cstr2_obj = dict_cstrs['cstr_%s'%cstr2][slot]

                mutually_exclusive_cols.append((cstr1_obj.col, cstr2_obj.col))

        return mutually_exclusive_cols


    def get_variabs(self):
        '''
        Collect all variables of this component.

        Return values:
            list of all variable symbols
        '''

        return [vv
                for var in self.VARIABS + self.VARIABS_TIME
                if hasattr(self, var)
                for vv in getattr(self, var).values()]


