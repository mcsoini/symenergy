#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch for sympy.solvers.solveset.linear_coeffs. This adds an initial checl
python ```
if isinstance(eq, sp.numbers.Float):
    c, terms = eq, tuple()
else:
    c, terms = _sympify(eq).as_coeff_add(*syms)
```
replacing the line
``` python
c, terms = _sympify(eq).as_coeff_add(*syms)
```


Issue: original function fails for pure sympy Float expressions.
See https://github.com/sympy/sympy/issues/17774

"""
from collections import defaultdict
from sympy.core.sympify import _sympify
from sympy.core import S, Add

def linear_coeffs(eq, *syms, **_kw):

    eq = _sympify(eq)
    if not eq.has(*syms):
        return [S.Zero]*len(syms) + [eq]
    c, terms = eq.as_coeff_add(*syms)

    d = defaultdict(list)
    d[0].extend(Add.make_args(c))
    for t in terms:
        m, f = t.as_coeff_mul(*syms)
        if len(f) != 1:
            break
        f = f[0]
        if f in syms:
            d[f].append(m)
        elif f.is_Add:
            d1 = linear_coeffs(f, *syms, **{'dict': True})
            d[0].append(m*d1.pop(0))
            for xf, vf in d1.items():
                d[xf].append(m*vf)
        else:
            break
    else:
        for k, v in d.items():
            d[k] = Add(*v)
        if not _kw:
            return [d.get(s, S.Zero) for s in syms] + [d[0]]
        return d  # default is still list but this won't matter
    raise ValueError('nonlinear term encountered: %s' % t)
