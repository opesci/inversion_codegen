from collections import Counter
from functools import singledispatch
from itertools import product

import sympy

from devito.symbolics import reuse_if_untouched
from devito.tools import as_mapper, flatten, split, timed_pass

__all__ = ['collect_derivatives']


@timed_pass()
def collect_derivatives(expressions):
    """
    Exploit linearity of finite-differences to collect `Derivative`'s of
    same type. This may help CIRE creating fewer temporaries while catching
    larger redundant sub-expressions.
    """
    processed = []
    for e in expressions:
        # Track type and number of nested Derivatives
        mapper = inspect(e)

        # E.g., 0.2*u.dx -> (0.2*u).dx
        ep = aggregate_coeffs(e, mapper)

        # E.g., (0.2*u).dx + (0.3*v).dx -> (0.2*u + 0.3*v).dx
        processed.append(factorize_derivatives(ep))

    return processed


# subpass: inspect

@singledispatch
def inspect(expr):
    mapper = {}
    counter = Counter()
    for a in expr.args:
        m = inspect(a)
        mapper.update(m)

        try:
            counter.update(m[a])
        except KeyError:
            pass

    mapper[expr] = counter

    return mapper


@inspect.register(sympy.Number)
@inspect.register(sympy.Symbol)
@inspect.register(sympy.Function)
def _(expr):
    return {}


@inspect.register(sympy.Derivative)
def _(expr):
    mapper = inspect(expr.expr)

    # Nested derivatives would reset the counting
    mapper[expr] = Counter([expr._metadata])

    return mapper


# subpass: aggregate_coeffs

# Note: in the recursion handlers below, `nn_derivs` stands for non-nested derivatives
# Its purpose is that of tracking *all* derivatives within a Derivative-induced scope
# For example, in `(...).dx`, the `.dx` derivative defines a new scope, and, in the
# `(...)` recursion handler, `nn_derivs` will carry information about all non-nested
# derivatives at any depth *inside* `(...)`


@singledispatch
def aggregate_coeffs(expr, mapper, nn_derivs=None):
    nn_derivs = nn_derivs or mapper.get(expr)

    args = [aggregate_coeffs(a, mapper, nn_derivs) for a in expr.args]
    expr = reuse_if_untouched(expr, args, evaluate=True)

    return expr


@aggregate_coeffs.register(sympy.Number)
@aggregate_coeffs.register(sympy.Symbol)
@aggregate_coeffs.register(sympy.Function)
def _(expr, mapper, nn_derivs=None):
    return expr


@aggregate_coeffs.register(sympy.Derivative)
def _(expr, mapper, nn_derivs=None):
    # Opens up a new derivative scope, so do not propagate `nn_derivs`
    args = [aggregate_coeffs(a, mapper) for a in expr.args]
    expr = reuse_if_untouched(expr, args)

    return expr


@aggregate_coeffs.register(sympy.Mul)
def _(expr, mapper, nn_derivs=None):
    nn_derivs = nn_derivs or mapper.get(expr)

    args = [aggregate_coeffs(a, mapper, nn_derivs) for a in expr.args]
    expr = reuse_if_untouched(expr, args)

    # Separate arguments containing derivatives from those which do not
    hope_coeffs = []
    with_derivs = []
    for a in args:
        if isinstance(a, sympy.Derivative):
            with_derivs.append((a, [a], []))
        else:
            derivs, others = split(a.args, lambda i: isinstance(i, sympy.Derivative))
            if a.is_Add and derivs:
                with_derivs.append((a, derivs, others))
            else:
                hope_coeffs.append(a)

    # E.g., non-linear term, expansion won't help (in fact, it would only
    # cause an increase in operation count), so we skip
    if len(with_derivs) > 1:
        return expr

    try:
        with_deriv, derivs, others = with_derivs.pop(0)
    except IndexError:
        # No derivatives found, give up
        return expr

    # Aggregating the potential coefficient won't help if, in the current scope
    # at least one derivative type does not appear more than once. In fact, aggregation
    # might even have a detrimental effect due to increasing the operation count by
    # expanding Muls), so we rather give if that's the case
    if not any(nn_derivs[i._metadata] > 1 for i in derivs):
        return expr

    # Is the potential coefficient really a coefficient?
    csymbols = set().union(*[i.free_symbols for i in hope_coeffs])
    cdims = [i._defines for i in csymbols if i.is_Dimension]
    ddims = [set(i.dims) for i in derivs]
    if any(i & j for i, j in product(cdims, ddims)):
        return expr

    # Redundancies unlikely to pop up along the time dimension
    if any(d.is_Time for d in flatten(ddims)):
        return expr

    if len(derivs) == 1 and with_deriv is derivs[0]:
        expr = with_deriv._new_from_self(expr=expr.func(*hope_coeffs, with_deriv.expr))
    else:
        others = [expr.func(*hope_coeffs, a) for a in others]
        derivs = [a._new_from_self(expr=expr.func(*hope_coeffs, a.expr)) for a in derivs]
        expr = with_deriv.func(*(derivs + others))

    return expr


# subpass: collect_derivatives

@singledispatch
def factorize_derivatives(expr):
    args = [factorize_derivatives(a) for a in expr.args]
    expr = reuse_if_untouched(expr, args)

    return expr


@factorize_derivatives.register(sympy.Number)
@factorize_derivatives.register(sympy.Symbol)
@factorize_derivatives.register(sympy.Function)
def _(expr):
    return expr


@factorize_derivatives.register(sympy.Add)
def _(expr):
    args = [factorize_derivatives(a) for a in expr.args]

    derivs, others = split(args, lambda a: isinstance(a, sympy.Derivative))
    if not derivs:
        return reuse_if_untouched(expr, args)

    # Map by type of derivative
    # Note: `D0(a) + D1(b) == D(a + b)` <=> `D0` and `D1`'s metadata match,
    # i.e. they are the same type of derivative
    mapper = as_mapper(derivs, lambda i: i._metadata)
    if len(mapper) == len(derivs):
        return expr

    args = list(others)
    for v in mapper.values():
        c = v[0]
        if len(v) == 1:
            args.append(c)
        else:
            args.append(c._new_from_self(expr=expr.func(*[i.expr for i in v])))
    expr = expr.func(*args)

    return expr
