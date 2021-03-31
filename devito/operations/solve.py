import sympy
from sympy.solvers.solveset import linear_coeffs

from devito.tools import as_tuple
from devito.types import Eq, Evaluable


__all__ = ['solve']


class Solve(Evaluable, sympy.Basic):  #TODO: inheriting from sympy.BAsic for sympify...
                                      # butmaybe there's a simpler way?

    def __init__(self, eq, target, **kwargs): 
        self.eq = eq
        self.target = target
        self.kwargs = kwargs

    def __repr__(self):
        return "Solve(%s, %s)" % (self.eq, self.target)

    __str__ = __repr__

    @property
    def args(self):
        return (self.eq,)

    @property
    def func(self):
        return lambda e, **kw: self.__class__(e, self.target, **self.kwargs)

    @property
    def evaluate(self):
        if isinstance(self.eq, Eq):
            expr = self.eq.lhs - self.eq.rhs if self.eq.rhs != 0 else self.eq.lhs
        else:
            expr = self.eq

        sols = []
        for e, t in zip(as_tuple(expr), as_tuple(self.target)):
            # Try first linear solver
            try:
                cc = linear_coeffs(e._eval_at(t).evaluate, t)
                sols.append(-cc[1]/cc[0])
            except ValueError:
                warning("Equation is not affine w.r.t the target, falling back to standard"
                       "sympy.solve that may be slow")
                self.kwargs['rational'] = False  # Avoid float indices
                self.kwargs['simplify'] = False  # Do not attempt premature optimisation
                sols.append(sympy.solve(e.evaluate, t, **self.kwargs)[0])

        # We need to rebuild the vector/tensor as sympy.solve outputs a tuple of solutions
        if len(sols) > 1:
            return target.new_from_mat(sols)
        else:
            return sols[0]


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
    return Solve(eq, target, **kwargs)
