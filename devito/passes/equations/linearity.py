from collections import namedtuple
from functools import singledispatch

import sympy

from devito.symbolics import q_leaf, q_function
from devito.tools import as_mapper, split, timed_pass

__all__ = ['collect_derivatives']


@timed_pass()
def collect_derivatives(expressions):
    """
    Exploit linearity of finite-differences to collect `Derivative`'s of
    same type. This may help CIRE by creating fewer temporaries and catching
    larger redundant sub-expressions.
    """
    processed = [_doit(e) for e in expressions]
    processed = list(zip(*processed))[0]
    return processed


Term = namedtuple('Term', 'other deriv func')
Term.__new__.__defaults__ = (None, None, None)

# `D0(a) + D1(b) == D(a + b)` <=> `D0` and `D1`'s metadata match, i.e. they
# are the same type of derivative
key = lambda e: e._metadata


@singledispatch
def _is_const_coeff(c, deriv):
    """True if coefficient definitely constant w.r.t. derivative, False otherwise."""
    return False


@_is_const_coeff.register(sympy.Number)
def _(c, deriv):
    return True


@_is_const_coeff.register(sympy.Symbol)
def _(c, deriv):
    try:
        return c.is_const
    except AttributeError:
        # Retrocompatibility -- if a sympy.Symbol, there's no `is_const` to query
        # We conservatively return False
        return False


@_is_const_coeff.register(sympy.Function)
def _(c, deriv):
    c_dims = set().union(*[getattr(i, '_defines', i) for i in c.free_symbols])
    deriv_dims = set().union(*[d._defines for d in deriv.dims])
    return not c_dims & deriv_dims


@_is_const_coeff.register(sympy.Expr)
def _(c, deriv):
    return all(_is_const_coeff(a, deriv) for a in c.args)


def _doit(expr):
    try:
        if q_function(expr) or q_leaf(expr):
            # Do not waste time
            return _doit_handle(expr, [])
    except AttributeError:
        # E.g., `Injection`
        return _doit_handle(expr, [])
    args = []
    terms = []
    for a in expr.args:
        ax, term = _doit(a)
        args.append(ax)
        terms.append(term)
    expr = expr.func(*args, evaluate=False)
    return _doit_handle(expr, terms)


@singledispatch
def _doit_handle(expr, terms):
    return expr, Term(expr)


@_doit_handle.register(sympy.Derivative)
def _(expr, terms):
    return expr, Term(sympy.S.One, expr)


@_doit_handle.register(sympy.Mul)
def _(expr, terms):
    derivs, others = split(terms, lambda i: i.deriv is not None)
    from IPython import embed; embed()
    if len(derivs) == 1:
        # Linear => propagate found Derivative upstream
        deriv = derivs[0].deriv
        other = expr.func(*[i.other for i in others])  # De-nest terms
        return expr, Term(other, deriv, expr.func)
    else:
        return expr, Term(expr)


@_doit_handle.register(sympy.Add)
def _(expr, terms):
    derivs, others = split(terms, lambda i: i.deriv is not None)
    if not derivs:
        return expr, Term(expr)

    # Map by type of derivative
    mapper = as_mapper(derivs, lambda i: key(i.deriv))
    if len(mapper) == len(derivs):
        return expr, Term(expr)

    processed = []
    for v in mapper.values():
        fact, nonfact = split(v, lambda i: _is_const_coeff(i.other, i.deriv))
        if fact:
            # Finally factorize derivative arguments
            func = fact[0].deriv._new_from_self
            exprs = []
            for i in fact:
                if i.func:
                    exprs.append(i.func(i.other, i.deriv.expr))
                else:
                    assert i.other == 1
                    exprs.append(i.deriv.expr)
            fact = [Term(func(expr=expr.func(*exprs)))]

        for i in fact + nonfact:
            if i.func:
                processed.append(i.func(i.other, i.deriv))
            else:
                processed.append(i.other)

    others = [i.other for i in others]
    expr = expr.func(*(processed + others))

    return expr, Term(expr)
