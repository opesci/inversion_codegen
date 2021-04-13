from collections import OrderedDict
from cached_property import cached_property

import numpy as np
from sympy.core.sympify import converter as sympify_converter

from devito.finite_differences import Differentiable
from devito.finite_differences.tools import make_shift_x0
from devito.types.basic import AbstractTensor
from devito.types.dense import Function, TimeFunction
from devito.types.utils import NODE

__all__ = ['TensorFunction', 'TensorTimeFunction', 'VectorFunction', 'VectorTimeFunction']


class TensorFunction(AbstractTensor):
    """
    Tensor valued Function represented as a Matrix.
    Each component is a Function or TimeFunction.

    A TensorFunction and the classes that inherit from it takes the same parameters as
    a DiscreteFunction and additionally:

    Parameters
    ----------
    name : str
        Name of the symbol.
    grid : Grid, optional
        Carries shape, dimensions, and dtype of the TensorFunction. When grid is not
        provided, shape and dimensions must be given. For MPI execution, a
        Grid is compulsory.
    space_order : int or 3-tuple of ints, optional
        Discretisation order for space derivatives. Defaults to 1. ``space_order`` also
        impacts the number of points available around a generic point of interest.  By
        default, ``space_order`` points are available on both sides of a generic point of
        interest, including those nearby the grid boundary. Sometimes, fewer points
        suffice; in other scenarios, more points are necessary. In such cases, instead of
        an integer, one can pass a 3-tuple ``(o, lp, rp)`` indicating the discretization
        order (``o``) as well as the number of points on the left (``lp``) and right
        (``rp``) sides of a generic point of interest.
    shape : tuple of ints, optional
        Shape of the domain region in grid points. Only necessary if ``grid`` isn't given.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if ``grid`` isn't given.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    staggered : Dimension or tuple of Dimension or Stagger, optional
        Define how the TensorFunction is staggered.
    initializer : callable or any object exposing the buffer interface, optional
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.
    padding : int or tuple of ints, optional
        .. deprecated:: shouldn't be used; padding is now automatically inserted.
        Allocate extra grid points to maximize data access alignment. When a tuple
        of ints, one int per Dimension should be provided.
    symmetric : bool, optional
        Whether the tensor is symmetric or not. Defaults to True.
    diagonal : Bool, optional
        Whether the tensor is diagonal or not. Defaults to False.
    staggered: tuple of Dimension, optional
        Staggering of each component, needs to have the size of the tensor. Defaults
        to the Dimensions.
    """

    _is_TimeDependent = False

    _sub_type = Function

    _class_priority = 10
    _op_priority = Differentiable._op_priority + 1.

    def __init_finalize__(self, *args, **kwargs):
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
        inds, _ = Function.__indices_setup__(grid=grid,
                                             dimensions=dimensions)
        self._space_dimensions = inds

    @classmethod
    def __subfunc_setup__(cls, *args, **kwargs):
        """
        Creates the components of the TensorFunction
        either from input or from input Dimensions.
        """
        comps = kwargs.get("components")
        if comps is not None:
            return comps

        grid = kwargs.get("grid")
        if grid is None:
            dims = kwargs.get('dimensions')
            if dims is None:
                raise TypeError("Need either `grid` or `dimensions`")
        else:
            dims = grid.dimensions
        stagg = kwargs.get("staggered", None)
        name = kwargs.get("name")
        symm = kwargs.get('symmetric', True)
        diag = kwargs.get('diagonal', False)

        funcs = []
        # Fill tensor, only upper diagonal if symmetric
        for i, d in enumerate(dims):
            funcs2 = [0 for _ in range(len(dims))]
            start = i if (symm or diag) else 0
            stop = i + 1 if diag else len(dims)
            for j in range(start, stop):
                kwargs["name"] = "%s_%s%s" % (name, d.name, dims[j].name)
                kwargs["staggered"] = (stagg[i][j] if stagg is not None
                                       else (NODE if i == j else (d, dims[j])))
                funcs2[j] = cls._sub_type(**kwargs)
            funcs.append(funcs2)

        # Symmetrize and fill diagonal if symmetric
        if symm:
            funcs = np.array(funcs) + np.triu(np.array(funcs), k=1).T
            funcs = funcs.tolist()
        return funcs

    def __getattr__(self, name):
        """
        Try calling a dynamically created FD shortcut.

        Notes
        -----
        This method acts as a fallback for __getattribute__
        """
        try:
            return self.applyfunc(lambda x: x if x == 0 else getattr(x, name))
        except:
            raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

    def _eval_at(self, func):
        """
        Evaluate tensor at func location
        """
        def entries(i, j, func):
            return getattr(self[i, j], '_eval_at', lambda x: self[i, j])(func[i, j])
        entry = lambda i, j: entries(i, j, func)
        return self._new(self.rows, self.cols, entry)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return Function.__indices_setup__(grid=kwargs.get('grid'),
                                          dimensions=kwargs.get('dimensions'))

    @property
    def _symbolic_functions(self):
        return frozenset().union(*[a._symbolic_functions for a in self.values()])

    @property
    def is_TimeDependent(self):
        return self._is_TimeDependent

    @property
    def space_dimensions(self):
        """Spatial dimensions."""
        return self._space_dimensions

    @cached_property
    def space_order(self):
        """The space order for all components."""
        return ({a.space_order for a in self} - {None}).pop()

    @property
    def is_diagonal(self):
        """Whether the tensor is diagonal."""
        return np.all([self[i, j] == 0 for j in range(self.cols)
                       for i in range(self.rows) if i != j])

    @property
    def evaluate(self):
        return self.applyfunc(lambda x: getattr(x, 'evaluate', x))

    def values(self):
        if self.is_diagonal:
            return [self[i, i] for i in range(self.shape[0])]
        elif self.is_symmetric:
            val = super(TensorFunction, self).values()
            return list(OrderedDict.fromkeys(val))
        else:
            return super(TensorFunction, self).values()

    def div(self, shift=None):
        """
        Divergence of the TensorFunction (is a VectorFunction).
        """
        comps = []
        func = vec_func(self)
        ndim = len(self.space_dimensions)
        shift_x0 = make_shift_x0(shift, (ndim, ndim))
        for i in range(len(self.space_dimensions)):
            comps.append(sum([getattr(self[j, i], 'd%s' % d.name)
                              (x0=shift_x0(shift, d, i, j))
                              for j, d in enumerate(self.space_dimensions)]))
        return func._new(comps)

    @property
    def laplace(self):
        """
        Laplacian of the TensorFunction.
        """
        comps = []
        func = vec_func(self)
        for j, d in enumerate(self.space_dimensions):
            comps.append(sum([getattr(self[j, i], 'd%s2' % d.name)
                              for i, d in enumerate(self.space_dimensions)]))
        return func._new(comps)

    def grad(self, shift=None):
        raise AttributeError("Gradient of a second order tensor not supported")

    def new_from_mat(self, mat):
        func = tens_func(self)
        return func._new(self.rows, self.cols, mat)

    def classof_prod(self, other, mat):
        try:
            is_mat = len(mat[0]) > 1
        except TypeError:
            is_mat = False
        is_time = (getattr(self, '_is_TimeDependent', False) or
                   getattr(other, '_is_TimeDependent', False))
        return mat_time_dict[(is_time, is_mat)]


class VectorFunction(TensorFunction):
    """
    Vector valued space varying Function as a rank 1 tensor of Function.
    """

    is_VectorValued = True
    is_TensorValued = False

    _sub_type = Function

    _is_TimeDependent = False

    @property
    def is_transposed(self):
        return self.shape[0] == 1

    @classmethod
    def __subfunc_setup__(cls, *args, **kwargs):
        """
        Creates the components of the VectorFunction
        either from input or from input dimensions.
        """
        comps = kwargs.get("components")
        if comps is not None:
            return comps
        funcs = []
        grid = kwargs.get("grid")
        if grid is None:
            dims = kwargs.get('dimensions')
            if dims is None:
                raise TypeError("Need either `grid` or `dimensions`")
        else:
            dims = grid.dimensions
        stagg = kwargs.get("staggered", None)
        name = kwargs.get("name")
        for i, d in enumerate(dims):
            kwargs["name"] = "%s_%s" % (name, d.name)
            kwargs["staggered"] = stagg[i] if stagg is not None else d
            funcs.append(cls._sub_type(**kwargs))

        return funcs

    # Custom repr and str
    def __str__(self):
        st = ''.join([' %-2s,' % c for c in self])[1:-1]
        return "Vector(%s)" % st

    __repr__ = __str__

    def div(self, shift=None):
        """
        Divergence of the VectorFunction, creates the divergence Function.
        """
        shift_x0 = make_shift_x0(shift, (len(self.space_dimensions),))
        return sum([getattr(self[i], 'd%s' % d.name)(x0=shift_x0(shift, d, None, i))
                    for i, d in enumerate(self.space_dimensions)])

    @property
    def laplace(self):
        """
        Laplacian of the VectorFunction, creates the Laplacian VectorFunction.
        """
        func = vec_func(self)
        comps = [sum([getattr(s, 'd%s2' % d.name) for d in self.space_dimensions])
                 for s in self]
        return func._new(comps)

    @property
    def curl(self):
        """
        Gradient of the (3D) VectorFunction, creates the curl VectorFunction.
        """

        if len(self.space_dimensions) != 3:
            raise AttributeError("Curl only supported for 3D VectorFunction")
        # The curl of a VectorFunction is a VectorFunction
        derivs = ['d%s' % d.name for d in self.space_dimensions]
        comp1 = getattr(self[2], derivs[1]) - getattr(self[1], derivs[2])
        comp2 = getattr(self[0], derivs[2]) - getattr(self[2], derivs[0])
        comp3 = getattr(self[1], derivs[0]) - getattr(self[0], derivs[1])

        func = vec_func(self)
        return func._new(3, 1, [comp1, comp2, comp3])

    def grad(self, shift=None):
        """
        Gradient of the VectorFunction, creates the gradient TensorFunction.
        """
        func = tens_func(self)
        ndim = len(self.space_dimensions)
        shift_x0 = make_shift_x0(shift, (ndim, ndim))
        comps = [[getattr(f, 'd%s' % d.name)(x0=shift_x0(shift, d, i, j))
                  for j, d in enumerate(self.space_dimensions)]
                 for i, f in enumerate(self)]
        return func._new(comps)

    def outer(self, other):
        comps = [[self[i] * other[j] for i in range(self.cols)] for j in range(self.cols)]
        func = tens_func(self, other)
        return func._new(comps)

    def new_from_mat(self, mat):
        func = vec_func(self)
        return func._new(self.rows, 1, mat)


class TensorTimeFunction(TensorFunction):
    """
    Time varying TensorFunction.
    """

    is_TensorValued = True

    _sub_type = TimeFunction

    _is_TimeDependent = True

    @cached_property
    def time_order(self):
        """The time order for all components."""
        return ({a.time_order for a in self} - {None}).pop()


class VectorTimeFunction(VectorFunction, TensorTimeFunction):
    """
    Time varying VectorFunction.
    """

    is_VectorValued = True
    is_TensorValued = False

    _sub_type = TimeFunction

    _is_TimeDependent = True
    _time_position = 0


mat_time_dict = {(True, True): TensorTimeFunction, (True, False): VectorTimeFunction,
                 (False, True): TensorFunction, (False, False): VectorFunction}


def vec_func(*funcs):
    return (VectorTimeFunction if any([getattr(f, 'is_TimeDependent', False)
            for f in funcs]) else VectorFunction)


def tens_func(*funcs):
    return (TensorTimeFunction if any([getattr(f, 'is_TimeDependent', False)
            for f in funcs]) else TensorFunction)


def sympify_tensor(arg):
    return arg


sympify_converter[TensorFunction] = sympify_tensor
