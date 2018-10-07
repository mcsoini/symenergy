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
                 if param_name in self.__dict__.keys()]

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
                     for cap_name, var_name in self.MAP_CAPACITY.items()
                     if var_name in self.__dict__.keys()
                         and cap_name in self.__dict__.keys()
                     for slot, var in getattr(self, var_name).items())

    def get_is_positive(self):
        '''
        Returns a tuple of all variables defined by the VARIABS_POSITIVE list.
        '''

        return tuple(var
                     for var_name in self.VARIABS_POSITIVE
                     if var_name in self.__dict__.keys()
                     for slot, var in getattr(self, var_name).items())



    def get_variabs(self):
        '''
        Collect all variables of this component.

        Return values:
            list of all variable symbols
        '''

        return [vv
                for var in self.VARIABS + self.VARIABS_TIME
                if var in self.__dict__.keys()
                for vv in getattr(self, var).values()]


