import sympy
from .differentiable import Differentiable

__all__ = ['cos', 'sin', 'tan', 'exp']


class cos(Differentiable, sympy.cos):
    pass


class sin(Differentiable, sympy.sin):
    pass


class tan(Differentiable, sympy.tan):
    pass


class exp(Differentiable, sympy.exp):
    pass


class log(Differentiable, sympy.log):
    pass
