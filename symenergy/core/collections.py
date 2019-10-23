#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: user
"""


import itertools
from orderedset import OrderedSet
from symenergy.core.parameter import Parameter
from symenergy.core.variable import Variable
from symenergy.core.constraint import Constraint
from sympy.core.symbol import Symbol




class AttributeCollection():
    '''
    Iterable collection of parameters/variables/constraints of components/model
    '''

    def __init__(self, name):

        self._name = name

        self._elements = []


    def append(self, el):

        type_args = type(el), str(self), type(el), self._expected_type
        assert isinstance(el, self._expected_type), ('Invalid type {} '
                         'appended to {}; got {}, expected {}').format(*type_args)

        self._elements.append(el)

        return el


    def tolist(self, return_attribute='', squeeze=True, unique=True, **kwargs):
        '''
        Return list of elements filtered by single element attribute.

        kwargs must be dict of type {element attribute: attribute value}.

        Example
        -------
        >>> `m.constraints.tolist(is_equality_constraint=True)`
        returns all equality constraints of model `m`
        >>> `m.constraints.tolist('col', is_equality_constraint=True)`
        returns the column names of all equality constraints of model `m`

        Parameters
        ----------
        return_attribute : str
            selected attribute of the element

        '''

        if isinstance(return_attribute, str):
            return_attribute = (return_attribute,)

        def get_retattr(el):

            retattr = tuple(getattr(el, attr, el) for attr in return_attribute)
            return retattr if retattr else el

        filt = lambda _: True  # case without filtering
        if kwargs:
            filt = lambda el: all(getattr(el, cond_attr) == val
                       for cond_attr, val in kwargs.items())

        return_list = [get_retattr(el) for el in self._elements if filt(el)]

        if unique:
            return_list = list(OrderedSet(return_list))

        if squeeze and len(return_attribute) == 1:
            return_list = list(itertools.chain.from_iterable(return_list))

        return return_list


    def copy(self):

        new = self.__class__(self._name)
        new._elements = self._elements.copy()

        return new


    def __radd__(self, othr):
        if othr == 0:
            return self
        else:
            return self.__add__(othr)


    def __add__(self, othr):

        tself = type(self)
        tothr = type(othr)

        if tself != tothr:
            raise TypeError('Trying to add %s and %s' % (tself, tothr))

        sum_ = self.__class__(name='model')  # added collections are model attributes
        sum_._elements = self._elements.copy() + othr._elements.copy()

        return sum_


    def __repr__(self):
        return '%s of %s'%(self.__class__.__name__, self._name)


    def to_dict(self, dict_struct={('name',): ''}, squeeze=True, **kwargs):
        '''
        Convert collection to arbitrarily nested dictionary.

        Returns a dictionary defined by the `dict_struct` parameter. This
        parameter is an arbitrarly nested dictionary with keys corresponding
        to tuples of element attribute names:
        `{('attr1', 'attr2'): {'attr3': {('attr4', 'attr5'): 'attr6'}}}`
        returns the values of `'attr6'` for all combinations of the other
        attributes.

        Example
        -------
        >>> `dict_struct={('base_name', 'comp_name'): {'slot': ''}}`



        If `dict_struct` is a string (name of element attribute), `to_dict`
        acts as a wrapper of
        :func:`symenergy.core.collections.AttributeCollection.tolist`

        Parameters
        ----------
        dict_struct : dict or str

        '''

        dict_struct = dict_struct.copy() if isinstance(dict_struct, dict) else dict_struct

        tuple_keys = not squeeze

        if isinstance(dict_struct, str):  # end of recursion
            return_attribute = dict_struct
            ret = self.tolist(return_attribute=return_attribute,
                              squeeze=squeeze, **kwargs)
            if len(ret) == 1 and squeeze:
                ret = ret[0]

        else:
            struct_key = list(dict_struct.keys())[0]
            if isinstance(struct_key, str):
                dict_struct = {(struct_key,): dict_struct[struct_key]}

            # select top level key
            unique_keys = OrderedSet(self.tolist(*dict_struct.keys(),
                                                 squeeze=False))

            dict_level = dict()
            for key_slct in unique_keys:
                # update kwargs with selected keys
                kwargs_all = {**kwargs, **dict(zip(*dict_struct.keys(), key_slct))}
                # recursive call for next-lower dict level and key value filter
                new_value = self.to_dict(dict_struct=dict_struct[list(dict_struct.keys())[0]],
                                         squeeze=squeeze,
                                         **kwargs_all)

                if not (isinstance(new_value, (tuple, list, set))
                        and not new_value):
                    new_key = (key_slct[0] if len(key_slct) == 1
                               and not tuple_keys else key_slct)
                    dict_level[new_key] = new_value

            ret = dict_level

        return ret

    def __call__(self, *args, **kwargs):

        return self.tolist(*args, **kwargs)



if __name__ == '__main__':


    dict_struct = {('mlt',): 'comp_name'}

    res_dict = m.constraints.to_dict(dict_struct=dict_struct)
    print('#' * 30)
    print(res_dict)

    res_dict


# %%



class ParameterCollection(AttributeCollection):
    '''
    Collection of type :class:`symenergy.core.parameter.Parameter`

    Note: This is the only `AttributeCollection` child which is not a
    collection of dicts.
    '''

    _expected_type = Parameter


class ConstraintCollection(AttributeCollection):
    '''
    Collection of type :class:`symenergy.core.constraint.Constraint`
    '''

    _expected_type = Constraint


class VariableCollection(AttributeCollection):
    '''
    Collection of type `dict(`:class:`sympy.core.symbol.Symbol``)`
    '''
    _expected_type = Variable


if __name__ == '__main__':
    pass
    m.constraints.get_filtered(is_equality_constraint=True)





