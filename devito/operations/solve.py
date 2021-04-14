from functools import singledispatch

from devito.finite_differences.differentiable import Add, Mul, EvalDerivative

__all__ = ['linsolve', 'SolveError']


class SolveError(Exception):
    """Base class for exceptions in this module."""
    pass


def linsolve(expr, target, **kwargs):
    """
    Linear solve for the targe in a single equation.

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


def eval_t(expr):
    """
    Evaluate all time derivatives in the expression.
    """
    try:
        assert any(d.is_Time for d in expr.dims)
        return expr.evaluate
    except:
        try:
            return expr.func(*[eval_t(a) for a in expr.args])
        except:
            return expr


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
