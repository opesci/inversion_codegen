from functools import singledispatch

import sympy

from devito.logger import warning
from devito.finite_differences.differentiable import Add, Mul, EvalDerivative
from devito.finite_differences.derivative import Derivative
from devito.tools import as_tuple

__all__ = ['solve', 'linsolve']


class SolveError(Exception):
    """Base class for exceptions in this module."""
    pass


def solve(eq, target, **kwargs):
    """
    Algebraically rearrange an Eq w.r.t. a given symbol.

    This is a wrapper around ``sympy.solve``.

    Parameters
    ----------
    eq : expr-like
        The equation to be rearranged.
    target : symbol
        The symbol w.r.t. which the equation is rearranged. May be a `Function`
        or any other symbolic object.
    **kwargs
        Symbolic optimizations applied while rearranging the equation. For more
        information. refer to ``sympy.solve.__doc__``.
    """
    try:
        eq = eq.lhs - eq.rhs if eq.rhs != 0 else eq.lhs
    except AttributeError:
        pass

    eqs, targets = as_tuple(eq), as_tuple(target)
    if len(eqs) == 0:
        warning("Empty input equation, returning `None`")
        return None

    sols = []
    for e, t in zip(eqs, targets):
        # Try first linear solver
        try:
            sols.append(linsolve(eval_time_derivatives(e), t))
        except SolveError:
            warning("Equation is not affine w.r.t the target, falling back to standard"
                    "sympy.solve that may be slow")
            kwargs['rational'] = False  # Avoid float indices
            kwargs['simplify'] = False  # Do not attempt premature optimisation
            sols.append(sympy.solve(e.evaluate, t, **kwargs)[0])

    # We need to rebuild the vector/tensor as sympy.solve outputs a tuple of solutions
    if len(sols) > 1:
        return target.new_from_mat(sols)
    else:
        return sols[0]


def linsolve(expr, target, **kwargs):
    """
    Linear solve for the target in a single equation.

    Parameters
    ----------
    expr : expr-like
        The expr to be rearranged.
    target : symbol
        The symbol w.r.t. which the equation is rearranged. May be a `Function`
        or any other symbolic object.
    """
    c = factorize_target(expr, target)
    if c != 0:
        return -expr.xreplace({target: 0})/c
    raise SolveError("No linear solution found")


@singledispatch
def eval_time_derivatives(expr):
    """
    Evaluate all time derivatives in the expression.
    """
    return expr


@eval_time_derivatives.register(Derivative)
def _(expr):
    if any(d.is_Time for d in expr.dims):
        return expr.evaluate
    return expr


@eval_time_derivatives.register(Add)
@eval_time_derivatives.register(Mul)
def _(expr):
    return expr.func(*[eval_time_derivatives(a) for a in expr.args])


@singledispatch
def factorize_target(expr, target):
    return 1 if expr is target else 0


@factorize_target.register(Add)
@factorize_target.register(EvalDerivative)
def _(expr, target):
    c = 0
    if not expr.has(target):
        return c

    for a in expr.args:
        c += factorize_target(a, target)
    return c


@factorize_target.register(Mul)
def _(expr, target):
    if not expr.has(target):
        return 0

    c = 1
    for a in expr.args:
        c *= a if not a.has(target) else factorize_target(a, target)
    return c
