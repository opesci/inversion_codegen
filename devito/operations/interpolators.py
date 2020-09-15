from abc import ABC, abstractmethod

import sympy
import numpy as np
from cached_property import cached_property

from devito.finite_differences import Evaluable
from devito.logger import warning
from devito.symbolics import retrieve_function_carriers, retrieve_functions, indexify, INT
from devito.tools import powerset, flatten, prod, as_list
from devito.types import Eq, Inc
from devito.types.basic import Scalar
from devito.types.dense import SubFunction
from devito.types.dimension import ConditionalDimension, Dimension, DefaultDimension

__all__ = ['LinearInterpolator', 'PrecomputedInterpolator']


class UnevaluatedSparseOperation(Evaluable):

    """
    Represents an Injection or an Interpolation operation performed on a
    SparseFunction. Evaluates to a list of Eq objects.

    Parameters
    ----------
    interpolator : Interpolator
        Interpolator object that will be used to evaluate the operation.
    callback : callable
        A routine generating the symbolic expressions for the operation.
    """

    subdomain = None

    def __init__(self, interpolator, callback):
        self.interpolator = interpolator
        self.callback = callback

    @property
    def evaluate(self):
        return_value = self.callback()
        assert(all(isinstance(i, Eq) for i in return_value))
        return return_value

    def __add__(self, other):
        return flatten([self, other])

    def __radd__(self, other):
        return flatten([other, self])


class Interpolation(UnevaluatedSparseOperation):

    """
    Represents an Interpolation operation performed on a SparseFunction.
    Evaluates to a list of Eq objects.
    """

    def __init__(self, expr, offset, increment, self_subs, interpolator, callback):
        # TODO: unused now, but will be necessary to compute the adjoint
        self.expr = expr
        self.offset = offset
        self.increment = increment
        self.self_subs = self_subs

        super(Interpolation, self).__init__(interpolator, callback)

    def __repr__(self):
        return "Interpolation(%s into %s)" % (repr(self.expr),
                                              repr(self.interpolator.sfunction))


class Injection(UnevaluatedSparseOperation):

    """
    Represents an Injection operation performed on a SparseFunction.
    Evaluates to a list of Eq objects.
    """

    def __init__(self, field, expr, offset, interpolator, callback):
        # TODO: unused now, but will be necessary to compute the adjoint
        self.field = field
        self.expr = expr
        self.offset = offset

        super(Injection, self).__init__(interpolator, callback)

    def __repr__(self):
        return "Injection(%s into %s)" % (repr(self.expr), repr(self.field))


class GenericInterpolator(ABC):

    """
    Abstract base class defining the interface for an interpolator.
    """

    @abstractmethod
    def inject(self, *args, **kwargs):
        pass

    @abstractmethod
    def interpolate(self, *args, **kwargs):
        pass


class LinearInterpolator(GenericInterpolator):

    """
    Concrete implementation of GenericInterpolator implementing a Linear interpolation
    scheme, i.e. Bilinear for 2D and Trilinear for 3D problems.

    Parameters
    ----------
    sfunction: The SparseFunction that this Interpolator operates on.
    """

    def __init__(self, sfunction):
        self.sfunction = sfunction

    @property
    def grid(self):
        return self.sfunction.grid

    @cached_property
    def _interpolation_coeffs(self):
        """
        Symbolic expression for the coefficients for sparse point interpolation
        according to:

            https://en.wikipedia.org/wiki/Bilinear_interpolation.

        Returns
        -------
        Matrix of coefficient expressions.
        """
        # Grid indices corresponding to the corners of the cell ie x1, y1, z1
        indices1 = tuple(sympy.symbols('%s1' % d) for d in self.grid.dimensions)
        indices2 = tuple(sympy.symbols('%s2' % d) for d in self.grid.dimensions)
        # 1, x1, y1, z1, x1*y1, ...
        indices = list(powerset(indices1))
        indices[0] = (1,)
        point_sym = list(powerset(self.sfunction._point_symbols))
        point_sym[0] = (1,)
        # 1, px. py, pz, px*py, ...
        A = []
        ref_A = [np.prod(ind) for ind in indices]
        # Create the matrix with the same increment order as the point increment
        for i in self.sfunction._point_increments:
            # substitute x1 by x2 if increment in that dimension
            subs = dict((indices1[d], indices2[d] if i[d] == 1 else indices1[d])
                        for d in range(len(i)))
            A += [[1] + [a.subs(subs) for a in ref_A[1:]]]

        A = sympy.Matrix(A)
        # Coordinate values of the sparse point
        p = sympy.Matrix([[np.prod(ind)] for ind in point_sym])

        # reference cell x1:0, x2:h_x
        left = dict((a, 0) for a in indices1)
        right = dict((b, dim.spacing) for b, dim in zip(indices2, self.grid.dimensions))
        reference_cell = {**left, **right}
        # Substitute in interpolation matrix
        A = A.subs(reference_cell)
        return A.inv().T * p

    def _interpolation_indices(self, variables, offset=0, field_offset=0):
        """
        Generate interpolation indices for the DiscreteFunctions in ``variables``.
        """
        index_matrix, points = self.sfunction._index_matrix(offset)

        idx_subs = []
        for i, idx in enumerate(index_matrix):
            # Introduce ConditionalDimension so that we don't go OOB
            mapper = {}
            for j, d in zip(idx, self.grid.dimensions):
                p = points[j]
                lb = sympy.And(p >= d.symbolic_min - self.sfunction._radius,
                               evaluate=False)
                ub = sympy.And(p <= d.symbolic_max + self.sfunction._radius,
                               evaluate=False)
                condition = sympy.And(lb, ub, evaluate=False)
                mapper[d] = ConditionalDimension(p.name, self.sfunction._sparse_dim,
                                                 condition=condition, indirect=True)

            # Track Indexed substitutions
            idx_subs.append(mapper)

        # Temporaries for the position
        temps = [Eq(v, k, implicit_dims=self.sfunction.dimensions)
                 for k, v in self.sfunction._position_map.items()]
        # Temporaries for the indirection dimensions
        temps.extend([Eq(v, k.subs(self.sfunction._position_map),
                         implicit_dims=self.sfunction.dimensions)
                      for k, v in points.items()])
        # Temporaries for the coefficients
        temps.extend([Eq(p, c.subs(self.sfunction._position_map),
                         implicit_dims=self.sfunction.dimensions)
                      for p, c in zip(self.sfunction._point_symbols,
                                      self.sfunction._coordinate_bases(field_offset))])

        return idx_subs, temps

    def interpolate(self, expr, offset=0, increment=False, self_subs={}):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        def callback():
            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr))

            # Need to get origin of the field in case it is staggered
            # TODO: handle each variable staggereing spearately
            field_offset = variables[0].origin
            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interpolation_indices(variables, offset,
                                                          field_offset=field_offset)

            # FIXME: Maybe needs a re-think wrt placement?
            # NOTE: Probably other better approaches available
            # Add idx_subs for `Function`'s on `SubDomain`'s
            add_subs = []
            fosd = set([f for f in retrieve_functions(expr, mode='unique')
                        if f.is_Function and f._subdomain])
            if len(fosd) > 1:
                raise NotImplementedError
            for f in fosd:
                dims = f._subdomain.dimensions
                for i in idx_subs:
                    add_subs.append({k: v.subs({d: v for d, v in zip(dims, i.values())})
                                     for k, v in f._subdomain._access_map.items()})
                idx_subs = add_subs

            # Substitute coordinate base symbols into the interpolation coefficients
            args = [_expr.xreplace(v_sub) * b.xreplace(v_sub)
                    for b, v_sub in zip(self._interpolation_coeffs, idx_subs)]

            # Accumulate point-wise contributions into a temporary
            rhs = Scalar(name='sum', dtype=self.sfunction.dtype)
            summands = [Eq(rhs, 0., implicit_dims=self.sfunction.dimensions)]
            summands.extend([Inc(rhs, i, implicit_dims=self.sfunction.dimensions)
                            for i in args])

            # Write/Incr `self`
            lhs = self.sfunction.subs(self_subs)
            last = [Inc(lhs, rhs)] if increment else [Eq(lhs, rhs)]

            return temps + summands + last

        return Interpolation(expr, offset, increment, self_subs, self, callback)

    def inject(self, field, expr, offset=0):
        """
        Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : expr-like
            Injected expression.
        offset : int, optional
            Additional offset from the boundary.
        """
        def callback():
            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr)) + [field]

            # Need to get origin of the field in case it is staggered
            field_offset = field.origin
            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interpolation_indices(variables, offset,
                                                          field_offset=field_offset)
            add_subs = []
            fosd = set([f for f in as_list(field) if f.is_Function and f._subdomain])
            if len(fosd) > 1:
                raise NotImplementedError
            for f in fosd:
                dims = f._subdomain.dimensions
                for i in idx_subs:
                    add_subs.append({k: v.subs({d: v for d, v in zip(dims, i.values())})
                                     for k, v in f._subdomain._access_map.items()})

            # Substitute coordinate base symbols into the interpolation coefficients
            if add_subs:
                eqns = [Inc(field.xreplace(asub), _expr.xreplace(vsub) * b,
                            implicit_dims=self.sfunction.dimensions)
                        for b, asub, vsub in zip(self._interpolation_coeffs,
                                                 add_subs, idx_subs)]
            else:
                eqns = [Inc(field.xreplace(vsub), _expr.xreplace(vsub) * b,
                            implicit_dims=self.sfunction.dimensions)
                        for b, vsub in zip(self._interpolation_coeffs, idx_subs)]

            return temps + eqns

        return Injection(field, expr, offset, self, callback)


class PrecomputedInterpolator(GenericInterpolator):

    def __init__(self, obj, r, gridpoints_data, coefficients_data):
        if not isinstance(r, int):
            raise TypeError('Need `r` int argument')
        if r <= 0:
            raise ValueError('`r` must be > 0')
        self.r = r
        self.obj = obj
        self._npoint = obj._npoint
        gridpoints = SubFunction(name="%s_gridpoints" % self.obj.name, dtype=np.int32,
                                 dimensions=(self.obj.indices[-1], Dimension(name='d')),
                                 shape=(self._npoint, self.obj.grid.dim), space_order=0,
                                 parent=self)

        assert(gridpoints_data is not None)
        gridpoints.data[:] = gridpoints_data[:]
        self.obj._gridpoints = gridpoints

        interpolation_coeffs = SubFunction(name="%s_interpolation_coeffs" % self.obj.name,
                                           dimensions=(self.obj.indices[-1],
                                                       Dimension(name='d'),
                                                       Dimension(name='i')),
                                           shape=(self.obj.npoint, self.obj.grid.dim,
                                                  self.r),
                                           dtype=self.obj.dtype, space_order=0,
                                           parent=self)
        assert(coefficients_data is not None)
        interpolation_coeffs.data[:] = coefficients_data[:]
        self.obj._interpolation_coeffs = interpolation_coeffs
        warning("Ensure that the provided interpolation coefficient and grid point " +
                "values are computed on the final grid that will be used for other " +
                "computations.")

    def interpolate(self, expr, offset=0, increment=False, self_subs={}):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        def callback():
            _expr = indexify(expr)

            p, _, _ = self.obj.interpolation_coeffs.indices
            dim_subs = []
            coeffs = []
            for i, d in enumerate(self.obj.grid.dimensions):
                rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
                dim_subs.append((d, INT(rd + self.obj.gridpoints[p, i])))
                coeffs.append(self.obj.interpolation_coeffs[p, i, rd])
            # Apply optional time symbol substitutions to lhs of assignment
            lhs = self.obj.subs(self_subs)
            rhs = prod(coeffs) * _expr.subs(dim_subs)

            return [Eq(lhs, lhs + rhs)]

        return Interpolation(expr, offset, increment, self_subs, self, callback)

    def inject(self, field, expr, offset=0):
        """
        Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : expr-like
            Injected expression.
        offset : int, optional
            Additional offset from the boundary.
        """
        def callback():
            _expr = indexify(expr)
            _field = indexify(field)

            p, _ = self.gridpoints.indices
            dim_subs = []
            coeffs = []
            for i, d in enumerate(self.grid.dimensions):
                rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
                dim_subs.append((d, INT(rd + self.gridpoints[p, i])))
                coeffs.append(self.obj.interpolation_coeffs[p, i, rd])
            rhs = prod(coeffs) * _expr
            _field = _field.subs(dim_subs)
            return [Eq(_field, _field + rhs.subs(dim_subs))]

        return Injection(field, expr, offset, self, callback)
