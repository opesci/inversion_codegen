from collections import OrderedDict
from collections.abc import Iterable

from cached_property import cached_property
import sympy

from devito.finite_differences.finite_difference import (generic_derivative,
                                                         first_derivative,
                                                         cross_derivative)
from devito.finite_differences.differentiable import Differentiable
from devito.finite_differences.tools import direct, transpose
from devito.tools import as_mapper, as_tuple, filter_ordered, frozendict
from devito.types.utils import DimensionTuple

__all__ = ['Derivative']


class Derivative(sympy.Derivative, Differentiable):

    """
    An unevaluated Derivative, which carries metadata (Dimensions,
    derivative order, etc) describing how the derivative will be expanded
    upon evaluation.

    Parameters
    ----------
    expr : expr-like
        Expression for which the Derivative is produced.
    dims : Dimension or tuple of Dimension
        Dimenions w.r.t. which to differentiate.
    fd_order : int or tuple of int, optional
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil. Defaults to 1.
    deriv_order: int or tuple of int, optional
        Derivative order. Defaults to 1.
    side : Side or tuple of Side, optional
        Side of the finite difference location, centered (at x), left (at x - 1)
        or right (at x +1). Defaults to ``centered``.
    transpose : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to ``direct``.
    subs : dict, optional
        Substitutions to apply to the finite-difference expression after evaluation.
    x0 : dict, optional
        Origin (where the finite-difference is evaluated at) for the finite-difference
        scheme, e.g. {x: x, y: y + h_y/2}.

    Examples
    --------
    Creation

    >>> from devito import Function, Derivative, Grid
    >>> grid = Grid((10, 10))
    >>> x, y = grid.dimensions
    >>> u = Function(name="u", grid=grid, space_order=2)
    >>> Derivative(u, x)
    Derivative(u(x, y), x)

    This can also be obtained via the differential shortcut

    >>> u.dx
    Derivative(u(x, y), x)

    You can also specify the order as a keyword argument

    >>> Derivative(u, x, deriv_order=2)
    Derivative(u(x, y), (x, 2))

    Or as a tuple

    >>> Derivative(u, (x, 2))
    Derivative(u(x, y), (x, 2))

    Once again, this can be obtained via shortcut notation

    >>> u.dx2
    Derivative(u(x, y), (x, 2))

    Derivative object are also callable to change default setup:

    >>> u.dx2(x0=x + x.spacing)
    Derivative(u(x, y), (x, 2))

    will create the second derivative at x=x + x.spacing. Accepted arguments for dynamic
    evaluation are `x0`, `fd_order` and `side`.
    """

    _state = ('expr', 'dims', 'side', 'fd_order', 'transpose', '_ppsubs', 'x0')
    _fd_priority = 3

    def __new__(cls, expr, *dims, **kwargs):
        if type(expr) == sympy.Derivative:
            raise ValueError("Cannot nest sympy.Derivative with devito.Derivative")
        if not isinstance(expr, Differentiable):
            raise ValueError("`expr` must be a Differentiable object")

        new_dims, orders, fd_o, var_count = cls._process_kwargs(expr, *dims, **kwargs)

        # Construct the actual Derivative object
        obj = Differentiable.__new__(cls, expr, *var_count)
        obj._dims = tuple(OrderedDict.fromkeys(new_dims))

        skip = kwargs.get('preprocessed', False) or obj.ndims == 1

        obj._fd_order = fd_o if skip else DimensionTuple(*fd_o, getters=obj._dims)
        obj._deriv_order = orders if skip else DimensionTuple(*orders, getters=obj._dims)
        obj._side = kwargs.get("side")
        obj._transpose = kwargs.get("transpose", direct)
        obj._ppsubs = as_tuple(frozendict(i) for i in kwargs.get("subs", []))
        obj._x0 = frozendict(kwargs.get('x0', {}))
        return obj

    @classmethod
    def _process_kwargs(cls, expr, *dims, **kwargs):
        """
        Process arguments for the construction of a Derivative
        """
        # Skip costly processing if constructiong from preprocessed
        if kwargs.get('preprocessed', False):
            fd_orders = kwargs.get('fd_order')
            deriv_orders = kwargs.get('deriv_order')
            if len(dims) == 1:
                dims = tuple([dims[0]]*deriv_orders)
            variable_count = [sympy.Tuple(s, dims.count(s))
                              for s in filter_ordered(dims)]
            return dims, deriv_orders, fd_orders, variable_count
        # Check `dims`. It can be a single Dimension, an iterable of Dimensions, or even
        # an iterable of 2-tuple (Dimension, deriv_order)
        if len(dims) == 0:
            raise ValueError("Expected Dimension w.r.t. which to differentiate")
        elif len(dims) == 1:
            if isinstance(dims[0], Iterable):
                # Iterable of Dimensions
                if len(dims[0]) != 2:
                    raise ValueError("Expected `(dim, deriv_order)`, got %s" % dims[0])
                orders = kwargs.get('deriv_order', dims[0][1])
                if dims[0][1] != orders:
                    raise ValueError("Two different values of `deriv_order`")
                new_dims = tuple([dims[0][0]]*dims[0][1])
            else:
                # Single Dimension
                orders = kwargs.get('deriv_order', 1)
                if isinstance(orders, Iterable):
                    orders = orders[0]
                new_dims = tuple([dims[0]]*orders)
        else:
            # Iterable of 2-tuple, e.g. ((x, 2), (y, 3))
            new_dims = []
            orders = []
            d_ord = kwargs.get('deriv_order', tuple([1]*len(dims)))
            for d, o in zip(dims, d_ord):
                if isinstance(d, Iterable):
                    new_dims.extend([d[0] for _ in range(d[1])])
                    orders.append(d[1])
                else:
                    new_dims.extend([d for _ in range(o)])
                    orders.append(o)
            new_dims = as_tuple(new_dims)
            orders = as_tuple(orders)

        # Finite difference orders depending on input dimension (.dt or .dx)
        fd_orders = kwargs.get('fd_order', tuple([expr.time_order if
                                                  getattr(d, 'is_Time', False) else
                                                  expr.space_order for d in dims]))
        if len(dims) == 1 and isinstance(fd_orders, Iterable):
            fd_orders = fd_orders[0]

        # SymPy expects the list of variable w.r.t. which we differentiate to be a list
        # of 2-tuple `(s, count)` where s is the entity to diff wrt and count is the order
        # of the derivative
        variable_count = [sympy.Tuple(s, new_dims.count(s))
                          for s in filter_ordered(new_dims)]
        return new_dims, orders, fd_orders, variable_count

    def __call__(self, x0=None, fd_order=None, side=None):
        if self.ndims == 1:
            _fd_order = fd_order or self._fd_order
            _side = side or self._side
            new_x0 = {self.dims[0]: x0} if x0 is not None else self.x0
            return self._new_from_self(fd_order=_fd_order, side=_side, x0=new_x0)

        if side is not None:
            raise TypeError("Side only supported for first order single"
                            "Dimension derivative such as `.dxl` or .dx(side=left)")
        # Cross derivative
        _x0 = dict(self._x0)
        _fd_order = dict(self.fd_order._getters)
        try:
            _fd_order.update(**(fd_order or {}))
            _fd_order = tuple(_fd_order.values())
            _fd_order = DimensionTuple(*_fd_order, getters=self.dims)
            _x0.update(x0)
        except AttributeError:
            raise TypeError("Multi-dimensional Derivative, input expected as a dict")

        return self._new_from_self(fd_order=_fd_order, x0=_x0)

    def _new_from_self(self, **kwargs):
        expr = kwargs.pop('expr', self.expr)
        _kwargs = {'deriv_order': self.deriv_order, 'fd_order': self.fd_order,
                   'side': self.side, 'transpose': self.transpose, 'subs': self._ppsubs,
                   'x0': self.x0, 'preprocessed': True}
        _kwargs.update(**kwargs)
        return Derivative(expr, *self.dims, **_kwargs)

    @property
    def func(self):
        return lambda *a, **kw: self._new_from_self(expr=a[0], **kw)

    def subs(self, *args, **kwargs):
        """
        Bypass sympy.Subs as Devito has its own lazy evaluation mechanism.
        """
        try:
            rules = dict(*args)
        except TypeError:
            rules = dict((args,))
        kwargs.pop('simultaneous', None)
        return self.xreplace(rules, **kwargs)

    def _xreplace(self, subs):
        """
        This is a helper method used internally by SymPy. We exploit it to postpone
        substitutions until evaluation.
        """
        subs = self._ppsubs + (subs,)  # Postponed substitutions
        return self._new_from_self(subs=subs), True

    @cached_property
    def _metadata(self):
        state = list(self._state)
        state.remove('expr')
        ret = [getattr(self, i) for i in state]
        ret.append(self.expr.staggered or (None,))
        return tuple(ret)

    @property
    def dims(self):
        return self._dims

    @property
    def ndims(self):
        return len(self._dims)

    @property
    def x0(self):
        return self._x0

    @property
    def fd_order(self):
        return self._fd_order

    @property
    def deriv_order(self):
        return self._deriv_order

    @property
    def side(self):
        return self._side

    @property
    def transpose(self):
        return self._transpose

    @property
    def is_TimeDependent(self):
        return self.expr.is_TimeDependent

    @property
    def T(self):
        """Transpose of the Derivative.

        FD derivatives can be represented as matrices and have adjoint/transpose.
        This is really useful for more advanced FD definitions. For example
        the conventional Laplacian is `.dxl.T * .dxl`
        """
        if self._transpose == direct:
            adjoint = transpose
        else:
            adjoint = direct

        return self._new_from_self(transpose=adjoint)

    def _eval_at(self, func):
        """
        Evaluates the derivative at the location of `func`. It is necessary for staggered
        setup where one could have Eq(u(x + h_x/2), v(x).dx)) in which case v(x).dx
        has to be computed at x=x + h_x/2.
        """
        # If an x0 already exists do not overwrite it
        x0 = self.x0 or dict(func.indices_ref._getters)
        if self.expr.is_Add:
            # If `expr` has both staggered and non-staggered terms such as
            # `(u(x + h_x/2) + v(x)).dx` then we exploit linearity of FD to split
            # it into `u(x + h_x/2).dx` and `v(x).dx`, since they require
            # different FD indices
            mapper = as_mapper(self.expr._args_diff, lambda i: i.staggered)
            args = [self.expr.func(*v) for v in mapper.values()]
            args.extend([a for a in self.expr.args if a not in self.expr._args_diff])
            args = [self._new_from_self(expr=a, x0=x0) for a in args]
            return self.expr.func(*args)
        elif self.expr.is_Mul:
            # For Mul, We treat the basic case `u(x + h_x/2) * v(x) which is what appear
            # in most equation with div(a * u) for example. The expression is re-centered
            # at the highest priority index (see _gather_for_diff) to compute the
            # derivative at x0.
            return self._new_from_self(x0=x0, expr=self.expr._gather_for_diff)
        else:
            # For every other cases, that has more functions or more complexe arithmetic,
            # there is not actual way to decide what to do so it’s as safe to use
            # the expression as is.
            return self._new_from_self(x0=x0)

    @property
    def evaluate(self):
        # Evaluate finite-difference.
        # Note: evaluate and _eval_fd splitted for potential future different
        # types of discretizations.
        return self._eval_fd(self.expr)

    @property
    def _eval_deriv(self):
        return self._eval_fd(self.expr)

    def _eval_fd(self, expr):
        """
        Evaluate finite difference approximation of the Derivative.
        Evaluation is carried out via the following four steps:

        - 1: Evaluate derivatives within the expression. For example given
            `f.dx * g`, `f.dx` will be evaluated first.
        - 2: Evaluate the finite difference for the (new) expression.
        - 3: Evaluate remaining terms (as `g` may need to be evaluated
        at a different point).
        - 4: Apply substitutions.
        """
        # Step 1: Evaluate derivatives within expression
        expr = getattr(expr, '_eval_deriv', expr)

        # Step 2: Evaluate FD of the new expression
        if self.side is not None and self.deriv_order == 1:
            res = first_derivative(expr, self.dims[0], self.fd_order,
                                   side=self.side, matvec=self.transpose,
                                   x0=self.x0)
        elif len(self.dims) > 1:
            res = cross_derivative(expr, self.dims, self.fd_order, self.deriv_order,
                                   matvec=self.transpose, x0=self.x0)
        else:
            res = generic_derivative(expr, *self.dims, self.fd_order, self.deriv_order,
                                     matvec=self.transpose, x0=self.x0)

        # Step 3: Evaluate remaining part of expression
        res = res.evaluate

        # Step 4: Apply substitution
        for e in self._ppsubs:
            res = res.xreplace(e)

        return res
