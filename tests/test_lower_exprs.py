import pytest
import numpy as np

from devito import (Grid, TimeFunction, SparseTimeFunction, Function, Operator, Eq,
                    SubDimension, SubDomain, configuration, solve)
from devito.finite_differences import Derivative
from devito.finite_differences.differentiable import diff2sympy
from devito.exceptions import InvalidOperator
from devito.ir import FindSymbols, retrieve_iteration_tree
from devito.passes.equations.linearity import collect_derivatives
from devito.tools import timed_region


class TestCollectDerivatives(object):

    """
    Test collect_derivatives and all mechanisms used by collect_derivatives
    indirectly.
    """

    def test_nocollection_if_diff_dims(self):
        """
        Test that expressions with different time dimensions are not collected.
        """
        grid = Grid((10,))

        f = TimeFunction(name="f", grid=grid, save=10)
        f2 = TimeFunction(name="f2", grid=grid, save=10)
        g = TimeFunction(name="g", grid=grid)
        g2 = TimeFunction(name="g2", grid=grid)
        w = Function(name="w", grid=grid)

        with timed_region('x'):
            eq = Eq(w, f.dt*g + f2.dt*g2)

            # Since all Function are time dependent, there should be no collection
            # and produce the same result as with the pre evaluated expression
            expr = Operator._lower_exprs([eq])[0]
            expr2 = Operator._lower_exprs([eq.evaluate])[0]

        assert expr == expr2

    def test_numeric_constant(self):
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)

        eq = Eq(u.forward, u.dx.dx + 0.3*u.dy.dx)
        leq = collect_derivatives.func([eq])[0]

        assert len(leq.find(Derivative)) == 3

    def test_symbolic_constant(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)

        eq = Eq(u.forward, u.dx.dx + dt**0.2*u.dy.dx)
        leq = collect_derivatives.func([eq])[0]

        assert len(leq.find(Derivative)) == 3

    def test_symbolic_constant_times_add(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)
        f = Function(name='f', grid=grid)

        eq = Eq(u.forward, u.laplace + dt**0.2*u.biharmonic(1/f))
        leq = collect_derivatives.func([eq])[0]

        assert len(eq.rhs.args) == 3
        assert len(leq.rhs.args) == 2
        assert all(isinstance(i, Derivative) for i in leq.rhs.args)

    def test_solve(self):
        """
        By remaining unevaluated until after Operator's collect_derivatives,
        the Derivatives after a solve() should be collected.
        """
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)

        pde = u.dt2 - (u.dx.dx + u.dy.dy) - u.dx.dy
        eq = Eq(u.forward, solve(pde, u.forward))
        leq = collect_derivatives.func([eq])[0]

        assert len(eq.rhs.find(Derivative)) == 5
        assert len(leq.rhs.find(Derivative)) == 4
        assert len(leq.rhs.args[2].find(Derivative)) == 3  # Check factorization

    def test_nocollection_if_unworthy(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing

        u = TimeFunction(name="u", grid=grid)

        eq = Eq(u.forward, (0.4 + dt)*(u.dx + u.dy))
        leq = collect_derivatives.func([eq])[0]

        assert eq == leq

    def test_pull_and_collect(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing
        hx, _ = grid.spacing_symbols

        u = TimeFunction(name="u", grid=grid)
        v = TimeFunction(name="v", grid=grid)

        eq = Eq(u.forward, ((0.4 + dt)*u.dx + 0.3)*hx + v.dx)
        leq = collect_derivatives.func([eq])[0]

        assert eq != leq
        args = leq.rhs.args
        assert len(args) == 2
        assert diff2sympy(args[0]) == 0.3*hx
        assert args[1] == (hx*(dt + 0.4)*u + v).dx

    def test_pull_and_collect_nested(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing
        hx, hy = grid.spacing_symbols

        u = TimeFunction(name="u", grid=grid, space_order=2)
        v = TimeFunction(name="v", grid=grid, space_order=2)

        eq = Eq(u.forward, (((0.4 + dt)*u.dx + 0.3)*hx + v.dx).dy + (0.2 + hy)*v.dy)
        leq = collect_derivatives.func([eq])[0]

        assert eq != leq
        assert leq.rhs == ((v + hx*(0.4 + dt)*u).dx + 0.3*hx + (0.2 + hy)*v).dy

    def test_pull_and_collect_nested_v2(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing
        hx, hy = grid.spacing_symbols

        u = TimeFunction(name="u", grid=grid, space_order=2)
        v = TimeFunction(name="v", grid=grid, space_order=2)

        eq = Eq(u.forward, ((0.4 + dt*(hy + 1. + hx*hy))*u.dx + 0.3)*hx + v.dx)
        leq = collect_derivatives.func([eq])[0]

        assert eq != leq
        assert leq.rhs == 0.3*hx + (hx*(0.4 + dt*(hy + 1. + hx*hy))*u + v).dx

    def test_pull_and_collect_nested_v3(self):
        grid = Grid(shape=(10, 10))
        dt = grid.time_dim.spacing
        hx, hy = grid.spacing_symbols

        a = Function(name="a", grid=grid, space_order=2)
        u = TimeFunction(name="u", grid=grid, space_order=2)
        v = TimeFunction(name="v", grid=grid, space_order=2)

        eq = Eq(u.forward, 0.4 + a*(hx + dt*(u.dx + v.dx)))
        leq = collect_derivatives.func([eq])[0]

        assert eq != leq
        assert leq.rhs == 0.4 + a*(hx + (dt*u + dt*v).dx)

    def test_nocollection_subdims(self):
        grid = Grid(shape=(10, 10))
        xi, yi = grid.interior.dimensions

        u = TimeFunction(name="u", grid=grid)
        v = TimeFunction(name="v", grid=grid)
        f = Function(name='f', grid=grid)

        eq = Eq(u.forward, u.dx + 0.2*f[xi, yi]*v.dx)
        leq = collect_derivatives.func([eq])[0]

        assert eq == leq

    def test_nocollection_staggered(self):
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions

        u = TimeFunction(name="u", grid=grid)
        v = TimeFunction(name="v", grid=grid, staggered=x)

        eq = Eq(u.forward, u.dx + v.dx)
        leq = collect_derivatives.func([eq])[0]

        assert eq == leq


class TestBuffering(object):

    def test_basic(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)

        eqn = Eq(u.forward, u + 1)

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 2
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1
        assert buffers.pop().symbolic_shape[0] == 2

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('async_degree', [2, 4])
    def test_async_degree(self, async_degree):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)

        eqn = Eq(u.forward, u + 1)

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('buffering', {'buf-async-degree': async_degree}))

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 2
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1
        assert buffers.pop().symbolic_shape[0] == async_degree

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

    def test_two_heterogeneous_buffers(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid, save=nt)
        v1 = TimeFunction(name='v', grid=grid, save=nt)

        eqns = [Eq(u.forward, u + v + 1),
                Eq(v.forward, u + v + v.backward)]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 3
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 2

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    def test_unread_buffered_function(self):
        nt = 10
        grid = Grid(shape=(4, 4))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid)
        v1 = TimeFunction(name='v', grid=grid)

        eqns = [Eq(v.forward, v + 1, implicit_dims=time),
                Eq(u, v)]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 1
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    def test_over_injection(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
        rec = SparseTimeFunction(name='rec', grid=grid, npoint=1, nt=nt)
        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2, save=nt)
        u1 = TimeFunction(name="u", grid=grid, time_order=2, space_order=2, save=nt)

        src.data[:] = 1.

        eqns = ([Eq(u.forward, u + 1)] +
                src.inject(field=u.forward, expr=src) +
                rec.interpolate(expr=u.forward))

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) ==\
            5 + bool(configuration['language'] != 'C')
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

    def test_over_one_subdomain(self):

        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 3, 3), y: ('middle', 3, 3)}

        s_d0 = sd0()
        nt = 10
        grid = Grid(shape=(10, 10), subdomains=(s_d0,))

        u = TimeFunction(name="u", grid=grid, save=nt)
        u1 = TimeFunction(name="u", grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid)
        v1 = TimeFunction(name='v', grid=grid)

        eqns = [Eq(v.forward, v + 1, subdomain=s_d0),
                Eq(u, v, subdomain=s_d0)]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt='buffering')

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    def test_over_two_subdomains_illegal(self):
        """
        Cannot use buffering when:

            * an Eq writes to `f` using one set of SubDimensions
            * another Eq reads from `f` through a different set of SubDimensions

        as the second Eq may want to read unwritten memory (i.e., zero-valued)
        in the buffered Function, while with buffering it might end up reading values
        written in a previous iteration, thus breaking a storage-related RAW dependence.
        """

        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 3, 3), y: ('middle', 3, 3)}

        class sd1(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 2, 2)}

        s_d0 = sd0()
        s_d1 = sd1()
        nt = 10
        grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1))

        u = TimeFunction(name="u", grid=grid, save=nt)

        eqns = [Eq(u.forward, u + 1, subdomain=s_d0),
                Eq(u.forward, u.forward + 1, subdomain=s_d1)]

        try:
            Operator(eqns, opt='buffering')
        except InvalidOperator:
            assert True
        except:
            assert False

    @pytest.mark.xfail(reason="Cannot deal with non-overlapping SubDimensions yet")
    def test_over_two_subdomains(self):

        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('left', 2), y: ('left', 2)}

        class sd1(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 2, 2)}

        s_d0 = sd0()
        s_d1 = sd1()
        nt = 10
        grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1))

        u = TimeFunction(name="u", grid=grid, save=nt)
        u1 = TimeFunction(name="u", grid=grid, save=nt)

        eqns = [Eq(u.forward, u + 1, subdomain=s_d0),
                Eq(u.forward, u.forward + u + 1, subdomain=s_d1)]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt='buffering')

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

    def test_subdimensions(self):
        nt = 10
        grid = Grid(shape=(10, 10, 10))
        x, y, z = grid.dimensions
        xi = SubDimension.middle(name='xi', parent=x, thickness_left=2, thickness_right=2)
        yi = SubDimension.middle(name='yi', parent=y, thickness_left=2, thickness_right=2)
        zi = SubDimension.middle(name='zi', parent=z, thickness_left=2, thickness_right=2)

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)

        eqn = Eq(u.forward, u + 1).xreplace({x: xi, y: yi, z: zi})

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 2
        assert len([i for i in FindSymbols().visit(op1) if i.is_Array]) == 1

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)
