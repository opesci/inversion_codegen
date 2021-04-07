import sympy
from sympy.solvers.solveset import linear_coeffs

from devito.logger import warning
from devito.tools import as_tuple
from devito.types import Eq, Evaluable


__all__ = ['solve']


class Solve(Evaluable, sympy.Basic):  #TODO: inheriting from sympy.BAsic for sympify...
                                      # butmaybe there's a simpler way?

    def __init__(self, expr, target, **kwargs):
        self.expr = expr
        self.target = target
        self.kwargs = kwargs

    def __repr__(self):
        return "Solve(%s, %s)" % (self.expr, self.target)

    __str__ = __repr__

    @property
    def args(self):
        return (self.expr,)

    @property
    def func(self):
        return lambda e, **kw: self.__class__(e, self.target, **self.kwargs)

    @property
    def evaluate(self):
        if isinstance(self.expr, Eq):
            expr = self.expr.lhs - self.expr.rhs if self.expr.rhs != 0 else self.expr.lhs
        else:
            expr = self.expr

        sols = []
        for e, t in zip(as_tuple(expr), as_tuple(self.target)):
            # Try first linear solver
            try:
                cc = linear_coeffs(e._eval_at(t).evaluate, t)
                sols.append(-cc[1]/cc[0])
            except ValueError:
                warning("Equation is not affine w.r.t the target, falling back to "
                        "standard sympy.solve that may be slow")
                self.kwargs['rational'] = False  # Avoid float indices
                self.kwargs['simplify'] = False  # Do not attempt premature optimisation
                sols.append(sympy.solve(e.evaluate, t, **self.kwargs)[0])

        # We need to rebuild the vector/tensor as sympy.solve outputs a tuple of solutions
        if len(sols) > 1:
            return self.target.new_from_mat(sols)
        else:
            return sols[0]


def solve(expr, target, **kwargs):
    """
    Algebraically rearrange an Eq w.r.t. a given symbol.

    This is a wrapper around ``sympy.solve``.

    Parameters
    ----------
    expr : expr-like
        The expression to be rearranged, possibly an Eq.
    target : symbol
        The symbol w.r.t. which the equation is rearranged. May be a `Function`
        or any other symbolic object.
    **kwargs
        Symbolic optimizations applied while rearranging the equation. For more
        information. refer to ``sympy.solve.__doc__``.
    """
    return Solve(expr, target, **kwargs)
