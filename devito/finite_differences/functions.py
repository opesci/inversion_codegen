import sympy
from .differentiable import Differentiable

__all__ = ['cos', 'sin', 'tan', 'exp']


class DifferentiableFunction(Differentiable, sympy.Function):
    pass


class cos(DifferentiableFunction, sympy.cos):
    pass


class sin(DifferentiableFunction, sympy.sin):
    pass


class tan(DifferentiableFunction, sympy.tan):
    pass


class exp(DifferentiableFunction, sympy.exp):
    pass

# TODO: Fix conflict with logger.log
# class log(DifferentiableFunction, sympy.log):
#     pass
