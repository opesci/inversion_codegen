from collections import namedtuple
from functools import singledispatch
from itertools import product

import sympy

from devito.symbolics import rebuild_if_untouched, q_leaf, q_function
from devito.tools import as_mapper, flatten, split, timed_pass

__all__ = ['collect_derivatives']


@timed_pass()
def collect_derivatives(expressions):
    """
    Exploit linearity of finite-differences to collect `Derivative`'s of
    same type. This may help CIRE by creating fewer temporaries and catching
    larger redundant sub-expressions.
    """
    # E.g., 0.2*u.dx -> (0.2*u).dx
    expressions = [_aggregate_coeffs(e).expr for e in expressions]

    # E.g., (0.2*u).dx + (0.3*v).dx -> (0.2*u + 0.3*v).dx
    processed = [_collect_derivatives(e) for e in expressions]
    processed = list(zip(*processed))[0]

    return processed


# subpass: aggregate_coeffs


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


class Bunch(object):

    def __init__(self, expr, derivs=None, dims=None):
        self.expr = expr
        self.derivs = derivs or []
        self.dims = dims or set()


@singledispatch
def _aggregate_coeffs(expr):
    args = [_aggregate_coeffs(a).expr for a in expr.args]
    return Bunch(rebuild_if_untouched(expr, args))


@_aggregate_coeffs.register(sympy.Add)
def _(expr):
    result = [_aggregate_coeffs(a) for a in expr.args]

    args = [i.expr for i in result]
    expr = rebuild_if_untouched(expr, args, evaluate=True)

    derivs = flatten(i.derivs for i in result)
    dims = set().union(*[i.dims for i in result])

    return Bunch(expr, derivs, dims)


@_aggregate_coeffs.register(sympy.Function)
def _(expr):
    dims = set()
    for i in expr.free_symbols:
        try:
            dims.update(i._defines)
        except AttributeError:
            pass
    return Bunch(expr, dims=dims)


@_aggregate_coeffs.register(sympy.Derivative)
def _(expr):
    result = [_aggregate_coeffs(a) for a in expr.args]

    args = [i.expr for i in result]
    expr = rebuild_if_untouched(expr, args)

    return Bunch(expr, [expr], result.dims)


@_aggregate_coeffs.register(sympy.Mul)
def _(expr):
    result = [_aggregate_coeffs(a) for a in expr.args]

    hope_coeffs = []
    with_derivs = []
    for i in result:
        if i.derivs:
            with_derivs.append((i.expr, i.derivs))
        else:
            hope_coeffs.append(i.expr)

    if not with_derivs:
        return Bunch(expr)
    elif len(with_derivs) > 1:
        # E.g., non-linear term, expansion won't help (in fact, it would only
        # cause an increase in operation count), so we skip
        return Bunch(expr)
    arg_deriv, derivs = with_derivs.pop(0)

    #TODO: Improve: just carry around the seen dims and compute intersection with .dx...
    cdims = set().union(*[i.dims for i in result])
    ddims 
    from IPython import embed; embed()
    if not all(_is_const_coeff(i, v) for i, v in product(hope_coeffs, derivs)):
        return Bunch(expr)

    if len(derivs) == 1 and arg_deriv is derivs[0]:
        expr = arg_deriv._new_from_self(expr=expr.func(*hope_coeffs, arg_deriv.expr))
    else:
        assert arg_deriv.is_Add
        args = []
        for a in arg_deriv.args:
            if a in derivs:
                args.append(a._new_from_self(expr=expr.func(*hope_coeffs, a.expr)))
            else:
                args.append(a)
        expr = arg_deriv.func(*args, evaluate=False)


    return Bunch(expr, dims=dims)


# subpass: collect_derivatives


Term = namedtuple('Term', 'other deriv func')
Term.__new__.__defaults__ = (None, None, None)

# `D0(a) + D1(b) == D(a + b)` <=> `D0` and `D1`'s metadata match, i.e. they
# are the same type of derivative
key = lambda e: e._metadata


def _collect_derivatives(expr):
    try:
        if q_function(expr) or q_leaf(expr):
            # Do not waste time
            return _collect_derivatives_handle(expr, [])
    except AttributeError:
        # E.g., `Injection`
        return _collect_derivatives_handle(expr, [])
    args = []
    terms = []
    for a in expr.args:
        ax, term = _collect_derivatives(a)
        args.append(ax)
        terms.append(term)
    expr = expr.func(*args, evaluate=False)
    return _collect_derivatives_handle(expr, terms)


@singledispatch
def _collect_derivatives_handle(expr, terms):
    return expr, Term(expr)


@_collect_derivatives_handle.register(sympy.Derivative)
def _(expr, terms):
    return expr, Term(sympy.S.One, expr)


@_collect_derivatives_handle.register(sympy.Mul)
def _(expr, terms):
    derivs, others = split(terms, lambda i: i.deriv is not None)
    if len(derivs) == 1:
        # Linear => propagate found Derivative upstream
        deriv = derivs[0].deriv
        other = expr.func(*[i.other for i in others])  # De-nest terms
        return expr, Term(other, deriv, expr.func)
    else:
        return expr, Term(expr)


@_collect_derivatives_handle.register(sympy.Add)
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
