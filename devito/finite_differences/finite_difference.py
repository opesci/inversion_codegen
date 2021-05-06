from sympy import sympify

from devito.finite_differences.differentiable import EvalDerivative
from devito.finite_differences.tools import (numeric_weights, symbolic_weights, left,
                                             right, generate_indices, centered, direct,
                                             transpose, check_input, check_symbolic)

__all__ = ['first_derivative', 'second_derivative', 'cross_derivative',
           'generic_derivative', 'left', 'right', 'centered', 'transpose',
           'generate_indices']

# Number of digits for FD coefficients to avoid roundup errors and non-deterministic
# code generation
_PRECISION = 9


@check_input
@check_symbolic
def first_derivative(expr, dim, fd_order=None, side=centered, matvec=direct,
                     symbolic=False, x0=None):
    """
    First-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the first-order derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int, optional
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil. Defaults to ``expr.space_order``
    side : Side, optional
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1). Defaults to ``centered``.
    matvec : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to ``direct``.
    x0 : dict, optional
        Origin of the finite-difference scheme as a map dim: origin_dim.

    Returns
    -------
    expr-like
        First-order derivative of ``expr``.

    Examples
    --------
    >>> from devito import Function, Grid, first_derivative, transpose
    >>> grid = Grid(shape=(4, 4))
    >>> x, _ = grid.dimensions
    >>> f = Function(name='f', grid=grid)
    >>> g = Function(name='g', grid=grid)
    >>> first_derivative(f*g, dim=x)
    -f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x

    Semantically, this is equivalent to

    >>> (f*g).dx
    Derivative(f(x, y)*g(x, y), x)

    The only difference is that in the latter case derivatives remain unevaluated.
    The expanded form is obtained via ``evaluate``

    >>> (f*g).dx.evaluate
    -f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x

    For the adjoint mode of the first derivative, pass ``matvec=transpose``

    >>> g = Function(name='g', grid=grid)
    >>> first_derivative(f*g, dim=x, matvec=transpose)
    -f(x, y)*g(x, y)/h_x + f(x - h_x, y)*g(x - h_x, y)/h_x

    This is also accessible via the .T shortcut

    >>> (f*g).dx.T.evaluate
    -f(x, y)*g(x, y)/h_x + f(x - h_x, y)*g(x - h_x, y)/h_x

    Finally the x0 argument allows to choose the origin of the finite-difference

    >>> first_derivative(f, dim=x, x0={x: 1})
    -f(1, y)/h_x + f(h_x + 1, y)/h_x
    """
    side = side
    order = fd_order or expr.space_order

    # Stencil positions for non-symmetric cross-derivatives with symmetric averaging
    ind = generate_indices(expr, dim, order, side=side, x0=x0)[0]

    # Finite difference weights from Taylor approximation with these positions
    if symbolic:
        c = symbolic_weights(expr, 1, ind, dim)
    else:
        c = numeric_weights(1, ind, dim)

    return indices_weights_to_fd(expr, dim, ind, c, matvec=matvec.val)


@check_input
@check_symbolic
def second_derivative(expr, dim, fd_order, **kwargs):
    """
    Second-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    stagger : Side, optional
        Shift of the finite-difference approximation.
    x0 : dict, optional
        Origin of the finite-difference scheme as a map dim: origin_dim.

    Returns
    -------
    expr-like
        Second-order derivative of ``expr``.

    Examples
    --------
    >>> from devito import Function, Grid, second_derivative
    >>> grid = Grid(shape=(4, 4))
    >>> x, _ = grid.dimensions
    >>> f = Function(name='f', grid=grid, space_order=2)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> second_derivative(f*g, dim=x, fd_order=2)
    -2.0*f(x, y)*g(x, y)/h_x**2 + f(x - h_x, y)*g(x - h_x, y)/h_x**2 +\
 f(x + h_x, y)*g(x + h_x, y)/h_x**2

    Semantically, this is equivalent to

    >>> (f*g).dx2
    Derivative(f(x, y)*g(x, y), (x, 2))

    The only difference is that in the latter case derivatives remain unevaluated.
    The expanded form is obtained via ``evaluate``

    >>> (f*g).dx2.evaluate
    -2.0*f(x, y)*g(x, y)/h_x**2 + f(x - h_x, y)*g(x - h_x, y)/h_x**2 +\
 f(x + h_x, y)*g(x + h_x, y)/h_x**2

    Finally the x0 argument allows to choose the origin of the finite-difference

    >>> second_derivative(f, dim=x, fd_order=2, x0={x: 1})
    -2.0*f(1, y)/h_x**2 + f(1 - h_x, y)/h_x**2 + f(h_x + 1, y)/h_x**2
    """

    return generic_derivative(expr, dim, fd_order, 2, **kwargs)


@check_input
@check_symbolic
def cross_derivative(expr, dims, fd_order, deriv_order, **kwargs):
    """
    Arbitrary-order cross derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the cross derivative is produced.
    dims : tuple of Dimension
        Dimensions w.r.t. which to differentiate.
    fd_order : tuple of ints
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    deriv_order : tuple of ints
        Derivative order, e.g. 2 for a second-order derivative.
    stagger : tuple of Side, optional
        Shift of the finite-difference approximation.
    x0 : dict, optional
        Origin of the finite-difference scheme as a map dim: origin_dim.

    Returns
    -------
    expr-like
        Cross-derivative of ``expr``.

    Examples
    --------
    >>> from devito import Function, Grid, second_derivative
    >>> grid = Grid(shape=(4, 4))
    >>> x, y = grid.dimensions
    >>> f = Function(name='f', grid=grid, space_order=2)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> cross_derivative(f*g, dims=(x, y), fd_order=(2, 2), deriv_order=(1, 1))
    (-1/h_y)*(-f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x) + \
(-f(x, y + h_y)*g(x, y + h_y)/h_x + f(x + h_x, y + h_y)*g(x + h_x, y + h_y)/h_x)/h_y

    Semantically, this is equivalent to

    >>> (f*g).dxdy
    Derivative(f(x, y)*g(x, y), x, y)

    The only difference is that in the latter case derivatives remain unevaluated.
    The expanded form is obtained via ``evaluate``

    >>> (f*g).dxdy.evaluate
    (-1/h_y)*(-f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x) + \
(-f(x, y + h_y)*g(x, y + h_y)/h_x + f(x + h_x, y + h_y)*g(x + h_x, y + h_y)/h_x)/h_y

    Finally the x0 argument allows to choose the origin of the finite-difference

    >>> cross_derivative(f*g, dims=(x, y), fd_order=(2, 2), deriv_order=(1, 1), \
    x0={x: 1, y: 2})
    (-1/h_y)*(-f(1, 2)*g(1, 2)/h_x + f(h_x + 1, 2)*g(h_x + 1, 2)/h_x) + (-f(1, h_y + 2)*\
g(1, h_y + 2)/h_x + f(h_x + 1, h_y + 2)*g(h_x + 1, h_y + 2)/h_x)/h_y
    """
    x0 = kwargs.get('x0', {})
    for d, fd, dim in zip(deriv_order, fd_order, dims):
        expr = generic_derivative(expr, dim=dim, fd_order=fd, deriv_order=d, x0=x0)

    return expr


@check_input
@check_symbolic
def generic_derivative(expr, dim, fd_order, deriv_order, symbolic=False,
                       matvec=direct, x0=None):
    """
    Arbitrary-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the derivative is produced.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    deriv_order : int
        Derivative order, e.g. 2 for a second-order derivative.
    stagger : Side, optional
        Shift of the finite-difference approximation.
    x0 : dict, optional
        Origin of the finite-difference scheme as a map dim: origin_dim.

    Returns
    -------
    expr-like
        ``deriv-order`` derivative of ``expr``.
    """
    # First order derivative with 2nd order FD is highly non-recommended so taking
    # first order fd that is a lot better
    if deriv_order == 1 and fd_order == 2 and not symbolic:
        fd_order = 1
    # Stencil positions
    indices, x0 = generate_indices(expr, dim, fd_order, x0=x0)

    # Finite difference weights from Taylor approximation with these positions
    if symbolic:
        c = symbolic_weights(expr, deriv_order, indices, x0)
    else:
        c = numeric_weights(deriv_order, indices, x0)

    return indices_weights_to_fd(expr, dim, indices, c, matvec=matvec.val)


def indices_weights_to_fd(expr, dim, inds, weights, matvec=1):
    """Expression from lists of indices and weights."""
    diff = dim.spacing

    d0 = ([d for d in expr.dimensions if d.root is dim] or [dim])[0]

    mapper = {dim: d0, diff: matvec*diff}

    # Loop through weights
    terms = []
    for i, c in zip(inds, weights):
        # Apply replacements to indices. Needs to be sympified in case
        # the indices are pure numbers
        try:
            iloc = i.xreplace(mapper)
        except AttributeError:
            iloc = sympify(i).xreplace(mapper)
        # Enforce fixed precision FD coefficients to avoid variations in results
        c = sympify(c).evalf(_PRECISION)
        terms.append(expr._subs(dim, iloc - (expr.indices_ref[dim] - dim)) * c)

    deriv = EvalDerivative(*terms, base=expr)

    return deriv
