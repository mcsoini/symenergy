#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deleting "simplify"

Original: solution = [simplify(sol).xreplace(swap) for sol in solution]
Patched:  solution = [(sol).xreplace(swap) for sol in solution]
"""

from types import GeneratorType
from sympy.core import S
from sympy.matrices import Matrix
from sympy.utilities import filldedent
from sympy.solvers.solvers import recast_to_symbols
from sympy.core.function import _mexpand
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.core.relational import Eq
from sympy.sets import FiniteSet

def linsolve(system, *symbols):

    if not system:
        return S.EmptySet

    # If second argument is an iterable
    if symbols and hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    sym_gen = isinstance(symbols, GeneratorType)

    swap = {}
    b = None  # if we don't get b the input was bad
    syms_needed_msg = None

    # unpack system

    if hasattr(system, '__iter__'):

        # 1). (A, b)
        if len(system) == 2 and isinstance(system[0], Matrix):
            A, b = system

        # 2). (eq1, eq2, ...)
        if not isinstance(system[0], Matrix):
            if sym_gen or not symbols:
                raise ValueError(filldedent('''
                    When passing a system of equations, the explicit
                    symbols for which a solution is being sought must
                    be given as a sequence, too.
                '''))
            system = [
                _mexpand(i.lhs - i.rhs if isinstance(i, Eq) else i,
                recursive=True) for i in system]
            system, symbols, swap = recast_to_symbols(system, symbols)
            A, b = linear_eq_to_matrix(system, symbols)
            syms_needed_msg = 'free symbols in the equations provided'

    elif isinstance(system, Matrix) and not (
            symbols and not isinstance(symbols, GeneratorType) and
            isinstance(symbols[0], Matrix)):
        # 3). A augmented with b
        A, b = system[:, :-1], system[:, -1:]

    if b is None:
        raise ValueError("Invalid arguments")

    syms_needed_msg  = syms_needed_msg or 'columns of A'

    if sym_gen:
        symbols = [next(symbols) for i in range(A.cols)]
        if any(set(symbols) & (A.free_symbols | b.free_symbols)):
            raise ValueError(filldedent('''
                At least one of the symbols provided
                already appears in the system to be solved.
                One way to avoid this is to use Dummy symbols in
                the generator, e.g. numbered_symbols('%s', cls=Dummy)
            ''' % symbols[0].name.rstrip('1234567890')))

    try:
        solution, params, free_syms = A.gauss_jordan_solve(b, freevar=True)
    except ValueError:
        # No solution
        return S.EmptySet

    # Replace free parameters with free symbols
    if params:
        if not symbols:
            symbols = [_ for _ in params]
            # re-use the parameters but put them in order
            # params       [x, y, z]
            # free_symbols [2, 0, 4]
            # idx          [1, 0, 2]
            idx = list(zip(*sorted(zip(free_syms, range(len(free_syms))))))[1]
            # simultaneous replacements {y: x, x: y, z: z}
            replace_dict = dict(zip(symbols, [symbols[i] for i in idx]))
        elif len(symbols) >= A.cols:
            replace_dict = {v: symbols[free_syms[k]] for k, v in enumerate(params)}
        else:
            raise IndexError(filldedent('''
                the number of symbols passed should have a length
                equal to the number of %s.
                ''' % syms_needed_msg))
        solution = [sol.xreplace(replace_dict) for sol in solution]

    solution = [(sol).xreplace(swap) for sol in solution]
    return FiniteSet(tuple(solution))
