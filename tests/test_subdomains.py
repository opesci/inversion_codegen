import pytest
import numpy as np
from math import floor

from devito import (Grid, Function, TimeFunction, Eq, solve, Operator, SubDomain,
                    SubDomainSet, Dimension, norm)
from devito.tools import timed_region
from examples.seismic import TimeAxis, RickerSource, Receiver


class TestSubdomains(object):
    """
    Class for testing SubDomains
    """

    def test_subdomain_dim(self):
        """
        Test that all dimensions including ones used as an expression
        are replaced by the subdimension dimensions.
        """
        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 6), y: ('middle', 1, 1)}
        s_d0 = sd0()
        grid = Grid(shape=(10, 10), subdomains=(s_d0,))
        x, y = grid.dimensions
        x1, y1 = s_d0.dimensions
        f = Function(name='f', grid=grid, dtype=np.int32)

        eq0 = Eq(f, x*f+y, subdomain=grid.subdomains['d0'])
        with timed_region('x'):
            expr = Operator._lower_exprs([eq0])[0]
        assert expr.rhs == x1 * f[x1 + 1, y1 + 1] + y1

    def test_multiple_middle(self):
        """
        Test Operator with two basic 'middle' subdomains defined.
        """
        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 6), y: ('middle', 1, 1)}
        s_d0 = sd0()

        class sd1(SubDomain):
            name = 'd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 6, 1), y: ('middle', 1, 1)}
        s_d1 = sd1()

        grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1))

        f = Function(name='f', grid=grid, dtype=np.int32)

        eq0 = Eq(f, f+1, subdomain=grid.subdomains['d0'])
        eq1 = Eq(f, f+2, subdomain=grid.subdomains['d1'])

        Operator([eq0, eq1])()

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        assert((np.array(f.data) == expected).all())

    def test_shape(self):
        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 6), y: ('middle', 1, 1)}
        s_d0 = sd0()

        class sd1(SubDomain):
            name = 'd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('right', 4), y: ('left', 2)}
        s_d1 = sd1()

        class sd2(SubDomain):
            name = 'd2'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('left', 3), y: ('middle', 1, 2)}
        s_d2 = sd2()

        grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1, s_d2))

        assert grid.subdomains['domain'].shape == (10, 10)
        assert grid.subdomains['interior'].shape == (8, 8)

        assert grid.subdomains['d0'].shape == (3, 8)
        assert grid.subdomains['d1'].shape == (4, 2)
        assert grid.subdomains['d2'].shape == (3, 7)

    def test_definitions(self):

        class sd0(SubDomain):
            name = 'sd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('right', 10)}

        class sd1(SubDomain):
            name = 'sd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('left', 10)}

        class sd2(SubDomain):
            name = 'sd2'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: y}

        class sd3(SubDomain):
            name = 'sd3'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 0, 0)}

        sd_def0 = sd0()
        sd_def1 = sd1()
        sd_def2 = sd2()
        sd_def3 = sd3()

        grid = Grid(shape=(10, 10), extent=(10, 10),
                    subdomains=(sd_def0, sd_def1, sd_def2, sd_def3))
        u0 = Function(name='u0', grid=grid)
        u1 = Function(name='u1', grid=grid)
        u2 = Function(name='u2', grid=grid)
        u3 = Function(name='u3', grid=grid)
        eq0 = Eq(u0, u0+1, subdomain=grid.subdomains['sd0'])
        eq1 = Eq(u1, u1+1, subdomain=grid.subdomains['sd1'])
        eq2 = Eq(u2, u2+1, subdomain=grid.subdomains['sd2'])
        eq3 = Eq(u3, u3+1, subdomain=grid.subdomains['sd3'])
        Operator([eq0, eq1, eq2, eq3])()

        assert u0.data.all() == u1.data.all() == u2.data.all() == u3.data.all()

    def test_iterate_NDomains(self):
        """
        Test that a set of subdomains are iterated upon correctly.
        """

        n_domains = 10

        class Inner(SubDomainSet):
            name = 'inner'

        bounds_xm = np.zeros((n_domains,), dtype=np.int32)
        bounds_xM = np.zeros((n_domains,), dtype=np.int32)
        bounds_ym = np.zeros((n_domains,), dtype=np.int32)
        bounds_yM = np.zeros((n_domains,), dtype=np.int32)

        for j in range(0, n_domains):
            bounds_xm[j] = j
            bounds_xM[j] = n_domains-1-j
            bounds_ym[j] = floor(j/2)
            bounds_yM[j] = floor(j/2)

        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        inner_sd = Inner(N=n_domains, bounds=bounds)

        grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))

        f = TimeFunction(name='f', grid=grid, dtype=np.int32)
        f.data[:] = 0

        stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
                     subdomain=grid.subdomains['inner'])

        op = Operator(stencil)
        op(time_m=0, time_M=9, dt=1)
        result = f.data[0]

        expected = np.zeros((10, 10), dtype=np.int32)
        for j in range(0, n_domains):
            expected[j, bounds_ym[j]:n_domains-bounds_yM[j]] = 10

        assert((np.array(result) == expected).all())

    def test_multi_eq(self):
        """
        Test SubDomainSet functionality when multiple equations are
        present.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        class MySubdomains(SubDomainSet):
            name = 'mydomains'

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = 1
        bounds_yM = 1
        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)
        my_sd = MySubdomains(N=n_domains, bounds=bounds)
        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd, ))

        assert(grid.subdomains['mydomains'].shape == ((3, 8), (3, 8)))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)
        h = Function(name='h', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+1, subdomain=grid.subdomains['mydomains'])
        eq2 = Eq(g, g+1)
        eq3 = Eq(h, h+2, subdomain=grid.subdomains['mydomains'])

        op = Operator([eq1, eq2, eq3])
        op.apply()

        expected1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        expected2 = np.full((10, 10), 1, dtype=np.int32)
        expected3 = 2*expected1

        assert((np.array(f.data) == expected1).all())
        assert((np.array(g.data) == expected2).all())
        assert((np.array(h.data) == expected3).all())

    def test_multi_sets(self):
        """
        Check functionality for when multiple subdomain sets are present.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        n = Dimension(name='n')
        m = Dimension(name='m')

        class MySubdomains1(SubDomainSet):
            name = 'mydomains1'
            implicit_dimension = n

        class MySubdomains2(SubDomainSet):
            name = 'mydomains2'
            implicit_dimension = m

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(1)
        bounds_yM = int(Ny/2+1)
        bounds1 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(Ny/2+1)
        bounds_yM = int(1)
        bounds2 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        my_sd1 = MySubdomains1(N=n_domains, bounds=bounds1)
        my_sd2 = MySubdomains2(N=n_domains, bounds=bounds2)

        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd1, my_sd2))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+1, subdomain=grid.subdomains['mydomains1'])
        eq2 = Eq(g, g+2, subdomain=grid.subdomains['mydomains2'])

        op = Operator([eq1, eq2])
        op.apply()

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

        assert((np.array(f.data[:]+g.data[:]) == expected).all())

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'overlap')])
    def test_subdomainset_mpi(self):

        n_domains = 5

        class Inner(SubDomainSet):
            name = 'inner'

        bounds_xm = np.zeros((n_domains,), dtype=np.int32)
        bounds_xM = np.zeros((n_domains,), dtype=np.int32)
        bounds_ym = np.zeros((n_domains,), dtype=np.int32)
        bounds_yM = np.zeros((n_domains,), dtype=np.int32)

        for j in range(0, n_domains):
            bounds_xm[j] = j
            bounds_xM[j] = j
            bounds_ym[j] = j
            bounds_yM[j] = 2*n_domains-1-j

        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        inner_sd = Inner(N=n_domains, bounds=bounds)

        grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))

        assert(grid.subdomains['inner'].shape == ((10, 1), (8, 1), (6, 1),
                                                  (4, 1), (2, 1)))

        f = TimeFunction(name='f', grid=grid, dtype=np.int32)
        f.data[:] = 0

        stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
                     subdomain=grid.subdomains['inner'])

        op = Operator(stencil)
        op(time_m=0, time_M=9, dt=1)
        result = f.data[0]

        fex = Function(name='fex', grid=grid)
        expected = np.zeros((10, 10), dtype=np.int32)
        for j in range(0, n_domains):
            expected[j, j:10-j] = 10
        fex.data[:] = np.transpose(expected)

        assert((np.array(result) == np.array(fex.data[:])).all())


class TestSubdomainFunctions(object):
    """
    Class for testing `Function`'s defined on `SubDomain`'s without MPI.
    """

    def test_basic_function(self):
        """
        Test a single `Function`
        """

        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 3, 1)}

        mid = Middle()

        grid = Grid(shape=(10, 10), extent=(9., 9.), subdomains=(mid, ))
        f = Function(name='f', grid=grid, subdomain=grid.subdomains['middle'])
        eq = Eq(f, f+1)

        assert(f.shape == grid.subdomains['middle'].shape)

        Operator(eq)()

        assert(np.all(f.data[:] == 1))

    def test_mixed_functions(self):
        """
        Test with one Function on a `SubDomain` and one not.
        """

        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 3, 1)}

        mid = Middle()

        grid = Grid(shape=(10, 10), extent=(9., 9.), subdomains=(mid, ))
        f = Function(name='f', grid=grid, subdomain=grid.subdomains['middle'])
        g = Function(name='g', grid=grid)

        assert(f.shape == grid.subdomains['middle'].shape)
        assert(g.shape == grid.shape)

        eq0 = Eq(f, g+f+1, subdomain=grid.subdomains['middle'])
        eq1 = Eq(g, 2*f, subdomain=grid.subdomains['middle'])
        eq2 = Eq(f, g+1, subdomain=grid.subdomains['middle'])

        Operator([eq0, eq1, eq2])()

        assert(np.all(f.data[:] == 3))
        assert(np.all(g.data[2:-2, 3:-1] == 2))


class TestSubdomainFunctionsParallel(object):
    """
    Class for testing `Function`'s defined on `SubDomain`'s with MPI.
    """

    @pytest.mark.parallel(mode=4)
    def test_mpi_function(self):
        """
        Test a single `Function`
        """

        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 3, 1)}

        mid = Middle()

        grid = Grid(shape=(10, 10), extent=(9., 9.), subdomains=(mid, ))
        f = Function(name='f', grid=grid, subdomain=grid.subdomains['middle'])
        eq = Eq(f, f+1)

        assert(f.shape == grid.subdomains['middle'].shape_local)

        Operator(eq)()

        assert(np.all(f.data[:] == 1))

    # FIXME: Fix this for MPI
    # @pytest.mark.parallel(mode=4)
    def test_acoustic_on_sd(self):

        class CompDom(SubDomain):

            name = 'comp_domain'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 20, 10), y: ('middle', 20, 10)}

        cdomain = CompDom()

        shape = (131, 131)
        extent = (1300, 1300)
        origin = (-200., -200.)

        v = np.empty(shape, dtype=np.float32)
        v[:, :71] = 1.5
        v[:, 71:] = 2.5

        grid = Grid(shape=shape, extent=extent, origin=origin, subdomains=(cdomain, ))

        t0 = 0.
        tn = 1000.
        dt = 1.6
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010
        src = RickerSource(name='src', grid=grid, f0=f0,
                           npoint=1, time_range=time_range)

        domain_size = np.array(extent)

        src.coordinates.data[0, :] = domain_size*.5
        src.coordinates.data[0, -1] = 20.

        rec = Receiver(name='rec', grid=grid, npoint=101, time_range=time_range)
        rec.coordinates.data[:, 0] = np.linspace(0, domain_size[0], num=101)
        rec.coordinates.data[:, 1] = 20.

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2,
                         subdomain=grid.subdomains['comp_domain'])
        m = Function(name='m', grid=grid)
        m.data[:] = 1./(v*v)

        # FIXME: should be pde = m * u.dt2 - u.laplace
        # but laplace not currently working
        pde = m * u.dt2 - (u.dx2 + u.dy2)
        stencil = Eq(u.forward, solve(pde, u.forward),
                     subdomain=grid.subdomains['comp_domain'])

        src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
        rec_term = rec.interpolate(expr=u.forward)

        op = Operator([stencil] + src_term + rec_term)

        op(time=time_range.num-1, dt=dt)

        # FIXME: Check why 1.e-5 fails on certain builds
        assert np.isclose(norm(rec), 436.3915, rtol=1.e-4)

    @pytest.mark.parallel(mode=4)
    def test_mixed_functions_mpi(self):
        """
        Test with one Function on a `SubDomain` and one not.
        """

        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 3, 1)}

        mid = Middle()

        grid = Grid(shape=(10, 10), extent=(9., 9.), subdomains=(mid, ))
        f = Function(name='f', grid=grid, subdomain=grid.subdomains['middle'])
        g = Function(name='g', grid=grid)

        assert(f.shape == grid.subdomains['middle'].shape_local)
        assert(g.shape == grid.shape_local)

        eq0 = Eq(f, g+f+1, subdomain=grid.subdomains['middle'])
        eq1 = Eq(g, 2*f, subdomain=grid.subdomains['middle'])
        eq2 = Eq(f, g+1, subdomain=grid.subdomains['middle'])

        Operator([eq0, eq1, eq2])()

        assert(np.all(f.data[:] == 3))
        assert(np.all(g.data[2:-2, 3:-1] == 2))

    def test_w_mixed_sparse(self):
        """
        Write a test when injecting onto a Function on a Sub and one not
        on a Sub.
        """
        # FIXME: Write this test
        assert(1 == 1)
