from collections import Counter, namedtuple
from functools import singledispatch
from itertools import product

import sympy

from devito.operations.solve import Solve
from devito.symbolics import rebuild_if_untouched
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
        ep = aggregate_coeffs(e, mapper).expr

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

Bunch = namedtuple('Bunch', 'expr derivs')
Bunch.__new__.__defaults__ = (None, None)


@singledispatch
def aggregate_coeffs(expr, mapper, nn_derivs=None):
    nn_derivs = nn_derivs or mapper.get(expr)

    args = [aggregate_coeffs(a, mapper, nn_derivs).expr for a in expr.args]
    expr = rebuild_if_untouched(expr, args)

    return Bunch(expr)


@aggregate_coeffs.register(sympy.Number)
@aggregate_coeffs.register(sympy.Symbol)
@aggregate_coeffs.register(sympy.Function)
def _(expr, mapper, nn_derivs=None):
    return Bunch(expr)


@aggregate_coeffs.register(sympy.Add)
def _(expr, mapper, nn_derivs=None):
    nn_derivs = nn_derivs or mapper.get(expr)

    result = [aggregate_coeffs(a, mapper, nn_derivs) for a in expr.args]
    args = [i.expr for i in result]
    expr = rebuild_if_untouched(expr, args, evaluate=True)

    derivs = flatten(i.derivs for i in result if i.derivs)

    return Bunch(expr, derivs)


@aggregate_coeffs.register(sympy.Derivative)
def _(expr, mapper, nn_derivs=None):
    result = [aggregate_coeffs(a, mapper) for a in expr.args]
    args = [i.expr for i in result]
    expr = rebuild_if_untouched(expr, args)

    return Bunch(expr, [expr])


@aggregate_coeffs.register(sympy.Mul)
def _(expr, mapper, nn_derivs=None):
    nn_derivs = nn_derivs or mapper.get(expr)

    result = [aggregate_coeffs(a, mapper, nn_derivs) for a in expr.args]

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

    # Aggregating the potential coefficient won't help if, in the current
    # scope, at least one derivative type does not appear more than once.
    # And, in fact, aggregation might even have a detrimental effect by
    # increasing the operation count due to Mul expansion), so we rather
    # give up w/ no structural changes to expr if that's the case
    if not any(nn_derivs[i._metadata] > 1 for i in derivs):
        return Bunch(expr)

    # Is the potential coefficient really a coefficient?
    csymbols = set().union(*[i.free_symbols for i in hope_coeffs])
    cdims = [i._defines for i in csymbols if i.is_Dimension]
    ddims = [set(i.dims) for i in derivs]
    if any(i & j for i, j in product(cdims, ddims)):
        return Bunch(expr)

    if len(derivs) == 1 and arg_deriv is derivs[0]:
        expr = arg_deriv._new_from_self(expr=expr.func(*hope_coeffs, arg_deriv.expr))
        derivs = [expr]
    else:
        assert arg_deriv.is_Add
        others = [expr.func(*hope_coeffs, a) for a in arg_deriv.args if a not in derivs]
        derivs = [a._new_from_self(expr=expr.func(*hope_coeffs, a.expr)) for a in derivs]
        expr = arg_deriv.func(*(derivs + others), evaluate=False)

    return Bunch(expr, derivs)


# subpass: collect_derivatives

@singledispatch
def factorize_derivatives(expr):
    return expr


@factorize_derivatives.register(sympy.Number)
@factorize_derivatives.register(sympy.Symbol)
@factorize_derivatives.register(sympy.Function)
def _(expr):
    return expr


@factorize_derivatives.register(sympy.Expr)
@factorize_derivatives.register(Solve)
def _(expr):
    args = [factorize_derivatives(a) for a in expr.args]
    expr = rebuild_if_untouched(expr, args)

    return expr


@factorize_derivatives.register(sympy.Add)
def _(expr):
    args = [factorize_derivatives(a) for a in expr.args]

    derivs, others = split(args, lambda a: isinstance(a, sympy.Derivative))
    if not derivs:
        return expr

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
