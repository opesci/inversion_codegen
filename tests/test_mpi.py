import numpy as np
import pytest
from cached_property import cached_property

from conftest import skipif
from devito import (Grid, Constant, Function, TimeFunction, SparseFunction,
                    SparseTimeFunction, Dimension, ConditionalDimension, SubDimension,
                    SubDomain, Eq, Ne, Inc, NODE, Operator, norm, inner, configuration,
                    switchconfig, generic_derivative)
from devito.data import LEFT, RIGHT
from devito.ir.iet import (Call, Conditional, Iteration, FindNodes, FindSymbols,
                           retrieve_iteration_tree)
from devito.mpi import MPI
from devito.mpi.routines import HaloUpdateCall
from examples.seismic.acoustic import acoustic_setup

pytestmark = skipif(['nompi'], whole_module=True)


class TestDistributor(object):

    @pytest.mark.parallel(mode=[2, 4])
    def test_partitioning(self):
        grid = Grid(shape=(15, 15))
        f = Function(name='f', grid=grid)

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(8, 15), (7, 15)],
            4: [(8, 8), (8, 7), (7, 8), (7, 7)]
        }
        assert f.shape == expected[distributor.nprocs][distributor.myrank]
        assert f.size_global == 225

    @pytest.mark.parallel(mode=[2, 4])
    def test_partitioning_fewer_dims(self):
        """Test domain decomposition for Functions defined over a strict subset
        of grid-decomposed dimensions."""
        size_x, size_y = 16, 16
        grid = Grid(shape=(size_x, size_y))
        x, y = grid.dimensions

        # A function with fewer dimensions that in `grid`
        f = Function(name='f', grid=grid, dimensions=(x,), shape=(size_x,))

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(8,), (8,)],
            4: [(8,), (8,), (8,), (8,)]
        }
        assert f.shape == expected[distributor.nprocs][distributor.myrank]

    @pytest.mark.parallel(mode=[2, 4])
    def test_partitioning_fewer_dims_timefunc(self):
        """Test domain decomposition for Functions defined over a strict subset
        of grid-decomposed dimensions."""
        size_x, size_y = 16, 16
        grid = Grid(shape=(size_x, size_y))
        x, y = grid.dimensions

        # A function with fewer dimensions that in `grid`
        f = TimeFunction(
            name='f',
            grid=grid,
            dimensions=(grid.time_dim, x,),
            shape=(10, size_x,),
        )

        distributor = grid.distributor
        expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
            2: [(8,), (8,)],
            4: [(8,), (8,), (8,), (8,)]
        }
        assert len(f.shape) == 2
        assert f.shape[0] == 10
        assert f.shape[1:] == expected[distributor.nprocs][distributor.myrank]

    @pytest.mark.parallel(mode=9)
    def test_neighborhood_horizontal_2d(self):
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions

        distributor = grid.distributor
        # Rank map:
        # ---------------y
        # | 0 | 1 | 2 |
        # -------------
        # | 3 | 4 | 5 |
        # -------------
        # | 6 | 7 | 8 |
        # -------------
        # |
        # x
        PN = MPI.PROC_NULL
        expected = {
            0: {x: {LEFT: PN, RIGHT: 3}, y: {LEFT: PN, RIGHT: 1}},
            1: {x: {LEFT: PN, RIGHT: 4}, y: {LEFT: 0, RIGHT: 2}},
            2: {x: {LEFT: PN, RIGHT: 5}, y: {LEFT: 1, RIGHT: PN}},
            3: {x: {LEFT: 0, RIGHT: 6}, y: {LEFT: PN, RIGHT: 4}},
            4: {x: {LEFT: 1, RIGHT: 7}, y: {LEFT: 3, RIGHT: 5}},
            5: {x: {LEFT: 2, RIGHT: 8}, y: {LEFT: 4, RIGHT: PN}},
            6: {x: {LEFT: 3, RIGHT: PN}, y: {LEFT: PN, RIGHT: 7}},
            7: {x: {LEFT: 4, RIGHT: PN}, y: {LEFT: 6, RIGHT: 8}},
            8: {x: {LEFT: 5, RIGHT: PN}, y: {LEFT: 7, RIGHT: PN}},
        }
        assert expected[distributor.myrank][x] == distributor.neighborhood[x]
        assert expected[distributor.myrank][y] == distributor.neighborhood[y]

    @pytest.mark.parallel(mode=9)
    def test_neighborhood_diagonal_2d(self):
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions

        distributor = grid.distributor
        # Rank map:
        # ---------------y
        # | 0 | 1 | 2 |
        # -------------
        # | 3 | 4 | 5 |
        # -------------
        # | 6 | 7 | 8 |
        # -------------
        # |
        # x
        PN = MPI.PROC_NULL
        expected = {
            0: {(LEFT, LEFT): PN, (LEFT, RIGHT): PN, (RIGHT, LEFT): PN, (RIGHT, RIGHT): 4},  # noqa
            1: {(LEFT, LEFT): PN, (LEFT, RIGHT): PN, (RIGHT, LEFT): 3, (RIGHT, RIGHT): 5},
            2: {(LEFT, LEFT): PN, (LEFT, RIGHT): PN, (RIGHT, LEFT): 4, (RIGHT, RIGHT): PN},  # noqa
            3: {(LEFT, LEFT): PN, (LEFT, RIGHT): 1, (RIGHT, LEFT): PN, (RIGHT, RIGHT): 7},
            4: {(LEFT, LEFT): 0, (LEFT, RIGHT): 2, (RIGHT, LEFT): 6, (RIGHT, RIGHT): 8},
            5: {(LEFT, LEFT): 1, (LEFT, RIGHT): PN, (RIGHT, LEFT): 7, (RIGHT, RIGHT): PN},
            6: {(LEFT, LEFT): PN, (LEFT, RIGHT): 4, (RIGHT, LEFT): PN, (RIGHT, RIGHT): PN},  # noqa
            7: {(LEFT, LEFT): 3, (LEFT, RIGHT): 5, (RIGHT, LEFT): PN, (RIGHT, RIGHT): PN},
            8: {(LEFT, LEFT): 4, (LEFT, RIGHT): PN, (RIGHT, LEFT): PN, (RIGHT, RIGHT): PN}  # noqa
        }
        assert all(expected[distributor.myrank][i] == distributor.neighborhood[i]
                   for i in [(LEFT, LEFT), (LEFT, RIGHT), (RIGHT, LEFT), (RIGHT, RIGHT)])

    @pytest.mark.parallel(mode=[2, 4])
    def test_ctypes_neighborhood(self):
        grid = Grid(shape=(4, 4))
        distributor = grid.distributor

        PN = MPI.PROC_NULL
        attrs = ['ll', 'lc', 'lr', 'cl', 'cc', 'cr', 'rl', 'rc', 'rr']
        expected = {  # nprocs -> [(rank0 xleft xright ...), (rank1 xleft ...), ...]
            2: [(PN, PN, PN, PN, 0, PN, PN, 1, PN),
                (PN, 0, PN, PN, 1, PN, PN, PN, PN)],
            4: [(PN, PN, PN, PN, 0, 1, PN, 2, 3),
                (PN, PN, PN, 0, 1, PN, 2, 3, PN),
                (PN, 0, 1, PN, 2, 3, PN, PN, PN),
                (0, 1, PN, 2, 3, PN, PN, PN, PN)]
        }

        mapper = dict(zip(attrs, expected[distributor.nprocs][distributor.myrank]))
        obj = distributor._obj_neighborhood
        value = obj._arg_defaults()[obj.name]
        assert all(getattr(value._obj, k) == v for k, v in mapper.items())

    @pytest.mark.parallel(mode=[4])
    def test_custom_topology(self):
        grid = Grid(shape=(15, 15))
        f = Function(name='f', grid=grid)

        # Default topology, computed by Devito
        distributor = grid.distributor
        assert distributor.topology == (2, 2)
        expected = [(8, 8), (8, 7), (7, 8), (7, 7)]
        assert f.shape == expected[distributor.myrank]
        assert f.size_global == 225

        # Now with a custom topology
        grid2 = Grid(shape=(15, 15), topology=(4, 1))
        f2 = Function(name='f', grid=grid2)

        distributor = grid2.distributor
        assert distributor.topology == (4, 1)
        expected = [(4, 15), (4, 15), (4, 15), (3, 15)]
        assert f2.shape == expected[distributor.myrank]
        assert f2.size_global == f.size_global


class TestFunction(object):

    @pytest.mark.parallel(mode=2)
    def test_halo_exchange_bilateral(self):
        """
        Test halo exchange between two processes organised in a 2x1 cartesian grid.

        On the left, the initial ``data_with_inhalo``; on the right, the situation
        after the halo exchange.

               rank0               rank0
            0 0 0 0 0 0         0 0 0 0 0 0
            0 1 1 1 1 0         0 1 1 1 1 0
            0 1 1 1 1 0         0 1 1 1 1 0
            0 1 1 1 1 0         0 1 1 1 1 0
            0 1 1 1 1 0         0 1 1 1 1 0
            0 0 0 0 0 0         0 2 2 2 2 0
                         ---->
               rank1               rank1
            0 0 0 0 0 0         0 1 1 1 1 0
            0 2 2 2 2 0         0 2 2 2 2 0
            0 2 2 2 2 0         0 2 2 2 2 0
            0 2 2 2 2 0         0 2 2 2 2 0
            0 2 2 2 2 0         0 2 2 2 2 0
            0 0 0 0 0 0         0 0 0 0 0 0
        """
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        f.data[:] = grid.distributor.myrank + 1

        # Now trigger a halo exchange...
        f.data_with_halo   # noqa

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[:] == 1.)
            assert np.all(f._data_ro_with_inhalo[-1, 1:-1] == 2.)
            assert np.all(f._data_ro_with_inhalo[0, :] == 0.)
        else:
            assert np.all(f.data_ro_domain[:] == 2.)
            assert np.all(f._data_ro_with_inhalo[0, 1:-1] == 1.)
            assert np.all(f._data_ro_with_inhalo[-1, :] == 0.)
        assert np.all(f._data_ro_with_inhalo[:, 0] == 0.)
        assert np.all(f._data_ro_with_inhalo[:, -1] == 0.)

    @pytest.mark.parallel(mode=2)
    def test_halo_exchange_bilateral_asymmetric(self):
        """
        Test halo exchange between two processes organised in a 2x1 cartesian grid.

        In this test, the size of left and right halo regions have different size.

        On the left, the initial ``data_with_inhalo``; on the right, the situation
        after the halo exchange.

                rank0                 rank0
            0 0 0 0 0 0 0         0 0 0 0 0 0 0
            0 1 1 1 1 0 0         0 1 1 1 1 0 0
            0 1 1 1 1 0 0         0 1 1 1 1 0 0
            0 1 1 1 1 0 0         0 1 1 1 1 0 0
            0 1 1 1 1 0 0         0 1 1 1 1 0 0
            0 0 0 0 0 0 0         0 2 2 2 2 0 0
            0 0 0 0 0 0 0         0 2 2 2 2 0 0
                           ---->
                rank1                 rank1
            0 0 0 0 0 0 0         0 1 1 1 1 0 0
            0 2 2 2 2 0 0         0 2 2 2 2 0 0
            0 2 2 2 2 0 0         0 2 2 2 2 0 0
            0 2 2 2 2 0 0         0 2 2 2 2 0 0
            0 2 2 2 2 0 0         0 2 2 2 2 0 0
            0 0 0 0 0 0 0         0 0 0 0 0 0 0
            0 0 0 0 0 0 0         0 0 0 0 0 0 0
        """
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid, space_order=(1, 1, 2))
        f.data[:] = grid.distributor.myrank + 1

        # Now trigger a halo exchange...
        f.data_with_halo   # noqa

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[:] == 1.)
            assert np.all(f._data_ro_with_inhalo[-2:, 1:-2] == 2.)
            assert np.all(f._data_ro_with_inhalo[0:1, :] == 0.)
        else:
            assert np.all(f.data_ro_domain[:] == 2.)
            assert np.all(f._data_ro_with_inhalo[:1, 1:-2] == 1.)
            assert np.all(f._data_ro_with_inhalo[-2:, :] == 0.)
        assert np.all(f._data_ro_with_inhalo[:, :1] == 0.)
        assert np.all(f._data_ro_with_inhalo[:, -2:] == 0.)

    @pytest.mark.parallel(mode=4)
    def test_halo_exchange_quadrilateral(self):
        """
        Test halo exchange between four processes organised in a 2x2 cartesian grid.

        The initial ``data_with_inhalo`` looks like:

               rank0           rank1
            0 0 0 0 0 0     0 0 0 0 0 0
            0 1 1 1 1 0     0 2 2 2 2 0
            0 1 1 1 1 0     0 2 2 2 2 0
            0 1 1 1 1 0     0 2 2 2 2 0
            0 1 1 1 1 0     0 2 2 2 2 0
            0 0 0 0 0 0     0 0 0 0 0 0

               rank2           rank3
            0 0 0 0 0 0     0 0 0 0 0 0
            0 3 3 3 3 0     0 4 4 4 4 0
            0 3 3 3 3 0     0 4 4 4 4 0
            0 3 3 3 3 0     0 4 4 4 4 0
            0 3 3 3 3 0     0 4 4 4 4 0
            0 0 0 0 0 0     0 0 0 0 0 0

        After the halo exchange, the following is expected and tested for:

               rank0           rank1
            0 0 0 0 0 0     0 0 0 0 0 0
            0 1 1 1 1 2     1 2 2 2 2 0
            0 1 1 1 1 2     1 2 2 2 2 0
            0 1 1 1 1 2     1 2 2 2 2 0
            0 1 1 1 1 2     1 2 2 2 2 0
            0 3 3 3 3 4     3 4 4 4 4 0

               rank2           rank3
            0 1 1 1 1 2     1 2 2 2 2 0
            0 3 3 3 3 4     3 4 4 4 4 0
            0 3 3 3 3 4     3 4 4 4 4 0
            0 3 3 3 3 4     3 4 4 4 4 0
            0 3 3 3 3 4     3 4 4 4 4 0
            0 0 0 0 0 0     0 0 0 0 0 0
        """
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        f.data[:] = grid.distributor.myrank + 1

        # Now trigger a halo exchange...
        f.data_with_halo   # noqa

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(f._data_ro_with_inhalo[0] == 0.)
            assert np.all(f._data_ro_with_inhalo[:, 0] == 0.)
            assert np.all(f._data_ro_with_inhalo[1:-1, -1] == 2.)
            assert np.all(f._data_ro_with_inhalo[-1, 1:-1] == 3.)
            assert f._data_ro_with_inhalo[-1, -1] == 4.
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(f._data_ro_with_inhalo[0] == 0.)
            assert np.all(f._data_ro_with_inhalo[:, -1] == 0.)
            assert np.all(f._data_ro_with_inhalo[1:-1, 0] == 1.)
            assert np.all(f._data_ro_with_inhalo[-1, 1:-1] == 4.)
            assert f._data_ro_with_inhalo[-1, 0] == 3.
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(f._data_ro_with_inhalo[-1] == 0.)
            assert np.all(f._data_ro_with_inhalo[:, 0] == 0.)
            assert np.all(f._data_ro_with_inhalo[1:-1, -1] == 4.)
            assert np.all(f._data_ro_with_inhalo[0, 1:-1] == 1.)
            assert f._data_ro_with_inhalo[0, -1] == 2.
        else:
            assert np.all(f._data_ro_with_inhalo[-1] == 0.)
            assert np.all(f._data_ro_with_inhalo[:, -1] == 0.)
            assert np.all(f._data_ro_with_inhalo[1:-1, 0] == 3.)
            assert np.all(f._data_ro_with_inhalo[0, 1:-1] == 2.)
            assert f._data_ro_with_inhalo[0, 0] == 1.

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('shape,expected', [
        ((15, 15), [((0, 8), (0, 8)), ((0, 8), (8, 15)),
                    ((8, 15), (0, 8)), ((8, 15), (8, 15))]),
    ])
    def test_local_indices(self, shape, expected):
        grid = Grid(shape=shape)
        f = Function(name='f', grid=grid)

        assert all(i == slice(*j)
                   for i, j in zip(f.local_indices, expected[grid.distributor.myrank]))


class TestSparseFunction(object):

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('shape, coords, points', [
        ((4, 4), ((1., 1.), (1., 3.), (3., 1.), (3., 3.)), 1),
        ((8, ), ((1.,), (3.,), (5.,), (7.,)), 1),
        ((8, ), ((1.,), (2.,), (3.,), (4.,), (5.,), (6.,), (7.,), (8.,)), 2)
    ])
    def test_ownership(self, shape, coords, points):
        """Given a sparse point ``p`` with known coordinates, this test checks
        that the MPI rank owning ``p`` is retrieved correctly."""
        grid = Grid(shape=shape, extent=shape)

        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)

        # The domain decomposition is so that the i-th MPI rank gets exactly one
        # sparse point `p` and, incidentally, `p` is logically owned by `i`
        assert len(sf.gridpoints) == points
        assert all(grid.distributor.glb_to_rank(i)[0] == grid.distributor.myrank
                   for i in sf.gridpoints)

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('coords,expected', [
        ([(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)], [[0.], [1.], [2.], [3.]]),
    ])
    def test_local_indices(self, coords, expected):
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        data = np.array([0., 1., 2., 3.])
        coords = np.array(coords)
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords))

        # Each of the 4 MPI ranks get one (randomly chosen) sparse point
        assert sf.npoint == 1

        sf.coordinates.data[:] = coords
        sf.data[:] = data

        expected = np.array(expected[grid.distributor.myrank])
        assert np.all(sf.data == expected)

    @pytest.mark.parallel(mode=4)
    def test_scatter_gather(self):
        """
        Test scattering and gathering of sparse data from and to a single MPI rank.

        The initial data distribution (A, B, C, and D are generic values) looks like:

               rank0           rank1           rank2           rank3
                [D]             [C]             [B]             [A]

        Logically (i.e., given point coordinates and domain decomposition), A belongs
        to rank0, B belongs to rank1, etc. Thus, after scattering, the data distribution
        is expected to be:

               rank0           rank1           rank2           rank3
                [A]             [B]             [C]             [D]

        Then, locally on each rank, a trivial *2 multiplication is performed:

               rank0           rank1           rank2           rank3
               [A*2]           [B*2]           [C*2]           [D*2]

        Finally, we gather the data values and we get:

               rank0           rank1           rank2           rank3
               [D*2]           [C*2]           [B*2]           [A*2]
        """
        grid = Grid(shape=(4, 4), extent=(4.0, 4.0))

        # Initialization
        data = np.array([3, 2, 1, 0])
        coords = np.array([(3., 3.), (3., 1.), (1., 3.), (1., 1.)])
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = data

        # Scatter
        loc_data = sf._dist_scatter()[sf]
        loc_coords = sf._dist_scatter()[sf.coordinates]
        assert len(loc_data) == 1
        assert loc_data[0] == grid.distributor.myrank
        # Do some local computation
        loc_data = loc_data*2

        # Gather
        sf._dist_gather(loc_data, loc_coords)
        assert len(sf.data) == 1
        assert np.all(sf.data == data[sf.local_indices]*2)

    @pytest.mark.parallel(mode=4)
    def test_sparse_coords(self):
        grid = Grid(shape=(21, 31, 21), extent=(20, 30, 20))
        x, y, z = grid.dimensions

        coords = np.zeros((21*31, 3))
        coords[:, 0] = np.asarray([i for i in range(21) for j in range(31)])
        coords[:, 1] = np.asarray([j for i in range(21) for j in range(31)])
        sf = SparseFunction(name="s", grid=grid, coordinates=coords, npoint=21*31)

        u = Function(name="u", grid=grid, space_order=1)
        u.data[:, :, 0] = np.reshape(np.asarray([i+j for i in range(21)
                                                 for j in range(31)]), (21, 31))

        op = Operator(sf.interpolate(u))
        op.apply()

        for i in range(21*31):
            coords_loc = sf.coordinates.data[i, 1]
            if coords_loc is not None:
                coords_loc += sf.coordinates.data[i, 0]
            assert sf.data[i] == coords_loc


class TestOperatorSimple(object):

    @pytest.mark.parallel(mode=[2, 4, 8])
    def test_trivial_eq_1d(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[t, x-1] + f[t, x+1] + 1))
        op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 3.)
        if f.grid.distributor.myrank == 0:
            assert f.data_ro_domain[0, 0] == 5.
            assert np.all(f.data_ro_domain[0, 1:] == 7.)
        elif f.grid.distributor.myrank == f.grid.distributor.nprocs - 1:
            assert f.data_ro_domain[0, -1] == 5.
            assert np.all(f.data_ro_domain[0, :-1] == 7.)
        else:
            assert np.all(f.data_ro_domain[0] == 7.)

    @pytest.mark.parallel(mode=[2])
    def test_trivial_eq_1d_asymmetric(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[t, x+1] + 1))
        op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 2.)
        if f.grid.distributor.myrank == 0:
            assert np.all(f.data_ro_domain[0] == 3.)
        else:
            assert np.all(f.data_ro_domain[0, :-1] == 3.)
            assert f.data_ro_domain[0, -1] == 2.

    @pytest.mark.parallel(mode=2)
    def test_trivial_eq_1d_save(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, save=5)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[time, x-1] + f[time, x+1] + 1))
        op.apply()

        time_M = op._prepare_arguments()['time_M']

        assert np.all(f.data_ro_domain[1] == 3.)
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[-1, time_M:] == 31.)
        else:
            assert np.all(f.data_ro_domain[-1, :-time_M] == 31.)

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag'), (4, 'overlap'),
                                (4, 'overlap2'), (4, 'diag2'), (4, 'full')])
    def test_trivial_eq_2d(self):
        grid = Grid(shape=(8, 8,))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid, space_order=1)
        f.data_with_halo[:] = 1.

        eqn = Eq(f.forward, f[t, x-1, y] + f[t, x+1, y] + f[t, x, y-1] + f[t, x, y+1])
        op = Operator(eqn)
        op.apply(time=1)

        # Expected computed values
        corner, side, interior = 10., 13., 16.

        glb_pos_map = f.grid.distributor.glb_pos_map
        assert np.all(f.data_ro_domain[0, 1:-1, 1:-1] == interior)
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert f.data_ro_domain[0, 0, 0] == corner
            assert np.all(f.data_ro_domain[0, 1:, :1] == side)
            assert np.all(f.data_ro_domain[0, :1, 1:] == side)
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert f.data_ro_domain[0, 0, -1] == corner
            assert np.all(f.data_ro_domain[0, :1, :-1] == side)
            assert np.all(f.data_ro_domain[0, 1:, -1:] == side)
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert f.data_ro_domain[0, -1, 0] == corner
            assert np.all(f.data_ro_domain[0, -1:, 1:] == side)
            assert np.all(f.data_ro_domain[0, :-1, :1] == side)
        else:
            assert f.data_ro_domain[0, -1, -1] == corner
            assert np.all(f.data_ro_domain[0, :-1, -1:] == side)
            assert np.all(f.data_ro_domain[0, -1:, :-1] == side)

    @pytest.mark.parallel(mode=[(8, 'basic'), (8, 'diag'), (8, 'overlap'),
                                (8, 'overlap2'), (8, 'diag2'), (8, 'full')])
    def test_trivial_eq_3d(self):
        grid = Grid(shape=(8, 8, 8))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid, space_order=1)
        f.data_with_halo[:] = 1.

        eqn = Eq(f.forward, (f[t, x-1, y, z] + f[t, x+1, y, z] +
                             f[t, x, y-1, z] + f[t, x, y+1, z] +
                             f[t, x, y, z-1] + f[t, x, y, z+1]))
        op = Operator(eqn)
        op.apply(time=1)

        # Expected computed values
        corner, side, face, interior = 21., 26., 31., 36.

        # Situation at t=0
        assert np.all(f.data_ro_domain[1] == 6.)
        # Situation at t=1
        # 1) corners
        for i in [[0, 0, 0], [0, 0, -1], [0, -1, 0], [0, -1, -1],
                  [-1, 0, 0], [-1, 0, -1], [-1, -1, 0], [-1, -1, -1]]:
            assert f.data_ro_domain[[0] + i] in [corner, None]
        # 2) sides
        for i in [[0, 0, slice(1, -1)], [0, -1, slice(1, -1)],
                  [0, slice(1, -1), 0], [0, slice(1, -1), -1],
                  [-1, 0, slice(1, -1)], [-1, -1, slice(1, -1)],
                  [-1, slice(1, -1), 0], [-1, slice(1, -1), -1],
                  [slice(1, -1), 0, 0], [slice(1, -1), 0, -1],
                  [slice(1, -1), -1, 0], [slice(1, -1), -1, -1]]:
            assert np.all(f.data_ro_domain[[0] + i] == side)
        # 3) faces
        for i in [[0, slice(1, -1), slice(1, -1)], [-1, slice(1, -1), slice(1, -1)],
                  [slice(1, -1), 0, slice(1, -1)], [slice(1, -1), -1, slice(1, -1)],
                  [slice(1, -1), slice(1, -1), 0], [slice(1, -1), slice(1, -1), -1]]:
            assert np.all(f.data_ro_domain[[0] + i] == face)
        # 4) interior
        assert np.all(f.data_ro_domain[0, 1:-1, 1:-1, 1:-1] == interior)

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag')])
    def test_multiple_eqs_funcs(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 0.
        g = TimeFunction(name='g', grid=grid)
        g.data_with_halo[:] = 0.

        op = Operator([Eq(f.forward, f[t, x+1] + g[t, x-1] + 1),
                       Eq(g.forward, f[t, x-1] + g[t, x+1] + 1)])
        op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 1.)
        if f.grid.distributor.myrank == 0:
            assert f.data_ro_domain[0, 0] == 2.
            assert np.all(f.data_ro_domain[0, 1:] == 3.)
        elif f.grid.distributor.myrank == f.grid.distributor.nprocs - 1:
            assert f.data_ro_domain[0, -1] == 2.
            assert np.all(f.data_ro_domain[0, :-1] == 3.)
        else:
            assert np.all(f.data_ro_domain[0] == 3.)

        # Also check that there are no redundant halo exchanges. Here, only
        # two are expected before the `x` Iteration, one for `f` and one for `g`
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

    @pytest.mark.parallel(mode=2)
    def test_reapply_with_different_functions(self):
        grid1 = Grid(shape=(30, 30, 30))
        f1 = Function(name='f', grid=grid1, space_order=4)

        op = Operator(Eq(f1, 1.))
        op.apply()

        grid2 = Grid(shape=(40, 40, 40))
        f2 = Function(name='f', grid=grid2, space_order=4)

        # Re-application
        op.apply(f=f2)

        assert np.all(f1.data == 1.)
        assert np.all(f2.data == 1.)


class TestCodeGeneration(object):

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_as_nostencil_basic(self):
        grid = Grid(shape=(12,))

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator([Eq(f.forward, f + 1.),
                       Eq(g, f + 1.)])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_as_nostencil_advanced(self):
        grid = Grid(shape=(4, 4))
        u = TimeFunction(name='u', grid=grid, space_order=4, time_order=2, save=None)
        v = TimeFunction(name='v', grid=grid, space_order=0, time_order=0, save=5)
        g = Function(name='g', grid=grid, space_order=0)
        i = Function(name='i', grid=grid, space_order=0)

        shift = Constant(name='shift', dtype=np.int32)

        step = Eq(u.forward, u - u.backward + 1)
        g_inc = Inc(g, u * v.subs(grid.time_dim, grid.time_dim - shift))
        i_inc = Inc(i, (v*v).subs(grid.time_dim, grid.time_dim - shift))

        op = Operator([step, g_inc, i_inc])

        # No stencil in the expressions, so no halo update required!
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_avoid_redundant_haloupdate(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        i = Dimension(name='i')
        j = Dimension(name='j')

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Inc(f[t+1, i], 1.),  # no halo update as it's an Inc
                       Eq(g, f[t, j] + 1)])  # access `f` at `t`, not `t+1`!

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_if_distr_but_sequential(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        # There is an anti-dependence between the first and second Eqs, so
        # the compiler places them in different x-loops. However, none of the
        # two loops should be preceded by a halo exchange, though for different
        # reasons:
        # * the equation in the first loop has no stencil
        # * the equation in the second loop is inherently sequential, so the
        #   compiler should be sufficiently smart to see that there is no point
        #   in adding a halo exchange
        op = Operator([Eq(f, f + 1),
                       Eq(f, f[t, x-1] + f[t, x+1] + 1.)])

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == 3
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_with_subdims(self):
        grid = Grid(shape=(4,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        thickness = 4

        u = TimeFunction(name='u', grid=grid, time_order=1)

        xleft = SubDimension.left(name='xleft', parent=x, thickness=thickness)
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)

        eq_centre = Eq(u[t+1, xi], u[t, xi-1] + u[t, xi+1] + 1.)
        eq_left = Eq(u[t+1, xleft], u[t+1, xleft+1] + u[t, xleft+1] + 1.)

        # There is only one halo update -- for the `eq_centre` expression.
        # `eq_left` gets no halo update since it's a left-SubDimension, which by
        # default (i.e., unless one passes `local=False` to SubDimension.left) is
        # assumed to be a local Dimension.
        op = Operator([eq_centre, eq_left])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    @pytest.mark.parallel(mode=1)
    def test_avoid_haloupdate_with_constant_index(self):
        grid = Grid(shape=(4,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)

        eq = Eq(u.forward, u[t, 1] + u[t, 1 + x.symbolic_min] + u[t, x])
        op = Operator(eq)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_do_haloupdate_with_constant_locindex(self):
        """
        Like `test_avoid_haloupdate_with_constant_index`, there is again
        a constant index, but this time along a loc-index (`t` Dimension),
        which needs to be handled by the `compute_loc_indices` function.
        The actual halo update is induced by u.dx.
        """
        grid = Grid(shape=(4,))
        x = grid.dimensions[0]

        u = TimeFunction(name='u', grid=grid)

        eq = Eq(u.forward, u[0, x] + u.dx)
        op = Operator(eq)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

    @pytest.mark.parallel(mode=1)
    def test_hoist_haloupdate_if_no_flowdep(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        i = Dimension(name='i')

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)

        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Inc(g[i], f[t, h[i]] + 1.)])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

        # Below, there is a flow-dependence along `x`, so a further halo update
        # before the Inc is required
        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Inc(g[i], f[t+1, h[i]] + 1.)])

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

    @pytest.mark.parallel(mode=1)
    def test_hoist_haloupdate_with_subdims(self):
        """
        This test stems from https://github.com/devitocodes/devito/issues/1119

        Ensure SubDimensions are treated just like any other Dimensions when it
        gets to placing halo exchanges.
        """
        grid = Grid(shape=(20, 20, 20))

        u = TimeFunction(name="u", grid=grid, space_order=2)
        U = TimeFunction(name="U", grid=grid, space_order=2)

        eqns = [Eq(u.forward, u.dx, subdomain=grid.interior),
                Eq(U.forward, U.dx + u.forward)]

        op = Operator(eqns)

        assert len(op._func_table) == 6

        # There are exactly two halo exchange calls in the Operator body
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2 + 8  # 8 are due to loop blocking
        assert calls[0].name == 'haloupdate0'
        assert calls[1].name == 'haloupdate0'

        # ... and none in the created efuncs
        calls = FindNodes(Call).visit(op._func_table['bf0'].root)
        assert len(calls) == 0
        calls = FindNodes(Call).visit(op._func_table['bf1'].root)
        assert len(calls) == 0

    @pytest.mark.parallel(mode=1)
    def test_hoist_haloupdate_from_innerloop(self):
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid, space_order=4)
        g = Function(name='g', grid=grid, space_order=2)

        eqns = [Eq(g, f.dzl + f.dzr + 1),
                Eq(f, g)]

        op = Operator(eqns, opt=('advanced', {'openmp': False}))

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

        # Also make sure the Call is at the right place in the IET
        assert op.body[-1].body[1].body[0].body[0].body[0].body[0].is_Call
        assert op.body[-1].body[1].body[0].body[0].body[1].is_Iteration

    @pytest.mark.parallel(mode=2)
    def test_unhoist_haloupdate_if_invariant(self):
        """
        Test an Operator that computes coupled equations in which the first
        one *does require* a halo update on a Dimension-invariant Function.
        """
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)

        f.data_with_halo[:] = 2.
        u.data_with_halo[:] = 1.

        eqns = [Eq(u.forward, u + f[x-1] + f[x+1] + 1.),
                Eq(f, u[t+1, x-1] + u[t+1, x+1] + 1.)]

        op = Operator(eqns)
        op.apply(time=1)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

        glb_pos_map = grid.distributor.glb_pos_map
        R = 1e-07
        if LEFT in glb_pos_map[x]:
            assert np.allclose(f.data_ro_domain[:5], [30., 56., 62., 67., 67.], rtol=R)
        else:
            assert np.allclose(f.data_ro_domain[5:], [67., 67., 62., 56., 30.], rtol=R)

    @pytest.mark.parallel(mode=[(2, 'basic'), (2, 'diag')])
    def test_redo_haloupdate_due_to_antidep(self):
        grid = Grid(shape=(12,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)

        op = Operator([Eq(f.forward, f[t, x-1] + f[t, x+1] + 1.),
                       Eq(g.forward, f[t+1, x-1] + f[t+1, x+1] + g)])
        op.apply(time=0)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

        assert np.all(f.data_ro_domain[1] == 1.)
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(g.data_ro_domain[1, 1:] == 2.)
        else:
            assert np.all(g.data_ro_domain[1, :-1] == 2.)

    @pytest.mark.parallel(mode=[(1, 'full')])
    def test_avoid_fullmode_if_crossloop_dep(self):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        f.data_with_halo[:] = 0.
        g.data_with_halo[:] = 1.

        op = Operator([Eq(f, g[x, y-1] + g[x, y+1]),
                       Eq(g, f)])

        # Exactly 4 routines will be generated for the basic mode
        assert len(op._func_table) == 4

        # Also check the numerical values
        op.apply()
        assert np.all(f.data[:] == 2.)

    @pytest.mark.parallel(mode=2)
    def test_avoid_haloudate_if_flowdep_along_other_dim(self):
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        xl = SubDimension.left(name='xl', parent=x, thickness=2)

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)

        f.data_with_halo[:] = 1.2
        g.data_with_halo[:] = 2.

        # Note: the subdomain is used to prevent the compiler from fusing the
        # third Eq in the first Eq's loop
        eqns = [Eq(f.forward, f[t, x-1] + f[t, x+1]),
                Eq(f[t+1, xl], f[t+1, xl] + 1.),
                Eq(g.forward, f[t, x-1] + f[t, x+1], subdomain=grid.interior)]

        op = Operator(eqns)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1

        op.apply(time_M=1)
        glb_pos_map = f.grid.distributor.glb_pos_map
        R = 1e-07  # Can't use np.all due to rounding error at the tails
        if LEFT in glb_pos_map[x]:
            assert np.allclose(f.data_ro_domain[0, :5], [5.6, 6.8, 5.8, 4.8, 4.8], rtol=R)
            assert np.allclose(g.data_ro_domain[0, :5], [2., 5.8, 5.8, 4.8, 4.8], rtol=R)
        else:
            assert np.allclose(f.data_ro_domain[0, 5:], [4.8, 4.8, 4.8, 4.8, 3.6], rtol=R)
            assert np.allclose(g.data_ro_domain[0, 5:], [4.8, 4.8, 4.8, 4.8, 2.], rtol=R)

    @pytest.mark.parallel(mode=2)
    def test_unmerge_haloupdate_if_no_locindices(self):
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = Function(name='f', grid=grid, space_order=1)
        g = TimeFunction(name='g', grid=grid, space_order=1, time_order=1)

        f.data_with_halo[:] = 0.
        g.data_with_halo[:] = 1.

        eqns = [Eq(f, g[t, x+1] + g[t, x-1]),
                Eq(g.forward, f[x-1] + f[x+1])]

        op = Operator(eqns)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

        titer = op.body[-1].body[0]
        assert titer.dim is grid.time_dim
        assert len(titer.nodes[0].body[0].body[0].body[0].body) == 1
        assert titer.nodes[0].body[0].body[0].body[0].body[0].is_Call
        parent = titer.nodes[0].body[0].body[0].body[1]
        if configuration['language'] == 'openmp':
            assert parent.body[0].body[0].is_Iteration
        else:
            assert parent.is_Iteration

        op.apply(time_M=1)

        glb_pos_map = f.grid.distributor.glb_pos_map
        R = 1e-07  # Can't use np.all due to rounding error at the tails
        if LEFT in glb_pos_map[x]:
            assert np.allclose(f.data_ro_domain[:5], [5., 6., 8., 8., 8.], rtol=R)
            assert np.allclose(g.data_ro_domain[0, :5], [6., 13., 14., 16., 16.], rtol=R)
        else:
            assert np.allclose(f.data_ro_domain[5:], [8., 8., 8., 6., 5.], rtol=R)
            assert np.allclose(g.data_ro_domain[0, 5:], [16., 16., 14., 13., 6.], rtol=R)

    @pytest.mark.parallel(mode=2)
    def test_unmerge_haloudate_if_diff_locindices(self):
        """
        In the Operator there are three Eqs:

        * the first one does *not* require a halo update
        * the second one requires a halo update for `f` at `t+1`
        * the third one requires a halo update for `f` at `t`

        Also:

        * the second and third Eqs cannot be fused in the same loop

        So in the IET we end up with two HaloSpots, one for the second and one
        for the third Eqs. These will *not* be merged because they operate at
        different `t`-indices (`t+1` and `t`).
        """
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid, space_order=2)
        g = TimeFunction(name='g', grid=grid)
        h = TimeFunction(name='h', grid=grid)

        f.data_with_halo[:] = 1.2
        g.data_with_halo[:] = 2.
        h.data_with_halo[:] = 3.1

        # Note: the subdomain is used to prevent the compiler from fusing the
        # third Eq in the second Eq's loop
        eqns = [Eq(f.forward, f + 1.),
                Eq(g.forward, f[t+1, x-1] + f[t+1, x+1]),
                Eq(h.forward, f[t, x-2] + f[t, x+2], subdomain=grid.interior)]

        op = Operator(eqns)

        calls = FindNodes(Call).visit(op)
        assert len(calls) == 2

        op.apply(time_M=1)
        glb_pos_map = f.grid.distributor.glb_pos_map
        R = 1e-07  # Can't use np.all due to rounding error at the tails
        if LEFT in glb_pos_map[x]:
            assert np.allclose(g.data_ro_domain[0, :5], [4.4, 6.4, 6.4, 6.4, 6.4], rtol=R)
            assert np.allclose(h.data_ro_domain[0, :5], [3.1, 3.4, 4.4, 4.4, 4.4], rtol=R)
        else:
            assert np.allclose(g.data_ro_domain[0, 5:], [6.4, 6.4, 6.4, 6.4, 4.4], rtol=R)
            assert np.allclose(h.data_ro_domain[0, 5:], [4.4, 4.4, 4.4, 3.4, 3.1], rtol=R)

    @pytest.mark.parallel(mode=1)
    @switchconfig(autopadding=True)
    def test_process_but_avoid_haloupdate_along_replicated(self):
        dx = Dimension(name='dx')
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions

        u = TimeFunction(name='u', grid=grid, space_order=4)
        c = Function(name='c', grid=grid, dimensions=(x, dx), shape=(10, 5))

        cases = [
            Eq(u.forward, (u.dx*c).dx + 1),
            Eq(u.forward, (u.dx*c[x, 0]).dx + 1)
        ]

        for eq in cases:
            op = Operator(eq, opt=('advanced', {'cire-mingain': 1}))

            calls = [i for i in FindNodes(Call).visit(op)
                     if isinstance(i, HaloUpdateCall)]
            assert len(calls) == 1
            assert calls[0].arguments[0] is u

    @pytest.mark.parallel(mode=1)
    def test_conditional_dimension(self):
        """
        Test the case of Functions in the condition of a ConditionalDimension.
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = TimeFunction(name='h', grid=grid, space_order=2)

        cd = ConditionalDimension(name='cd', parent=x, condition=~Ne(g, h[t, x+1, y]))

        eqns = [Eq(f.forward, f + 1, implicit_dims=cd),
                Eq(g, h + 1)]

        op = Operator(eqns)

        # No halo update here because the `x` Iteration is SEQUENTIAL
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 0

    @pytest.mark.parametrize('expr,expected', [
        ('f[t,x-1,y] + f[t,x+1,y]', {'rc', 'lc'}),
        ('f[t,x,y-1] + f[t,x,y+1]', {'cr', 'cl'}),
        ('f[t,x-1,y-1] + f[t,x,y+1]', {'cr', 'rr', 'rc', 'cl'}),
        ('f[t,x-1,y-1] + f[t,x+1,y+1]', {'cr', 'rr', 'rc', 'cl', 'll', 'lc'}),
    ])
    @pytest.mark.parallel(mode=[(1, 'diag')])
    def test_diag_comm_scheme(self, expr, expected):
        """
        Check that the 'diag' mode does not generate more communications
        than strictly necessary.
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions  # noqa
        t = grid.stepping_dim  # noqa

        f = TimeFunction(name='f', grid=grid)  # noqa

        op = Operator(Eq(f.forward, eval(expr)), opt=('advanced', {'openmp': False}))

        calls = FindNodes(Call).visit(op._func_table['haloupdate0'])
        destinations = {i.arguments[-2].field for i in calls}
        assert destinations == expected

    @pytest.mark.parallel(mode=[(1, 'full')])
    def test_poke_progress(self):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)

        eqn = Eq(f.forward, f[t, x-1, y] + f[t, x+1, y] + f[t, x, y-1] + f[t, x, y+1])
        op = Operator(eqn, opt=('advanced', {'par-dynamic-work': 0}))

        trees = retrieve_iteration_tree(op._func_table['compute0'].root)
        assert len(trees) == 2
        tree = trees[0]
        # Make sure `pokempi0` is the last node within the outer Iteration
        assert len(tree) == 2
        assert len(tree.root.nodes) == 2
        call = tree.root.nodes[1]
        assert call.name == 'pokempi0'
        assert call.arguments[0].name == 'msg0'
        if configuration['language'] == 'openmp':
            # W/ OpenMP, we prod until all comms have completed
            assert call.then_body[0].body[0].is_While
            # W/ OpenMP, we expect dynamic thread scheduling
            assert 'dynamic,1' in tree.root.pragmas[0].value
        else:
            # W/o OpenMP, it's a different story
            assert call._single_thread

        # Now we do as before, but enforcing loop blocking (by default off,
        # as heuristically it is not enabled when the Iteration nest has depth < 3)
        op = Operator(eqn, opt=('advanced', {'blockinner': True, 'par-dynamic-work': 0}))
        trees = retrieve_iteration_tree(op._func_table['bf0'].root)
        assert len(trees) == 2
        tree = trees[1]
        # Make sure `pokempi0` is the last node within the inner Iteration over blocks
        assert len(tree) == 2
        assert len(tree.root.nodes[0].nodes) == 2
        call = tree.root.nodes[0].nodes[1]
        assert call.name == 'pokempi0'
        assert call.arguments[0].name == 'msg0'
        if configuration['language'] == 'openmp':
            # W/ OpenMP, we prod until all comms have completed
            assert call.then_body[0].body[0].is_While
            # W/ OpenMP, we expect dynamic thread scheduling
            assert 'dynamic,1' in tree.root.pragmas[0].value
        else:
            # W/o OpenMP, it's a different story
            assert call._single_thread


class TestOperatorAdvanced(object):

    @pytest.mark.parallel(mode=4)
    def test_injection_wodup(self):
        """
        Test injection operator when the sparse points don't need to be replicated
        ("wodup" -> w/o duplication) over multiple MPI ranks.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        f = Function(name='f', grid=grid, space_order=0)
        f.data[:] = 0.
        coords = np.array([(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)])
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = 4.

        # This is the situation at this point
        # O is a grid point
        # * is a sparse point
        #
        # O --- O --- O --- O
        # |  *  |     |  *  |
        # O --- O --- O --- O
        # |     |     |     |
        # O --- O --- O --- O
        # |  *  |     |  *  |
        # O --- O --- O --- O

        op = Operator(sf.inject(field=f, expr=sf + 1))
        op.apply()

        assert np.all(f.data == 1.25)

    @pytest.mark.parallel(mode=4)
    def test_injection_wodup_wtime(self):
        """
        Just like ``test_injection_wodup``, but using a SparseTimeFunction
        instead of a SparseFunction. Hence, the data scattering/gathering now
        has to correctly pack/unpack multidimensional arrays.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        save = 3
        f = TimeFunction(name='f', grid=grid, save=save, space_order=0)
        f.data[:] = 0.
        coords = np.array([(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)])
        sf = SparseTimeFunction(name='sf', grid=grid, nt=save,
                                npoint=len(coords), coordinates=coords)
        sf.data[0, :] = 4.
        sf.data[1, :] = 8.
        sf.data[2, :] = 12.

        op = Operator(sf.inject(field=f, expr=sf + 1))
        op.apply()

        assert np.all(f.data[0] == 1.25)
        assert np.all(f.data[1] == 2.25)
        assert np.all(f.data[2] == 3.25)

    @pytest.mark.parallel(mode=4)
    def test_injection_dup(self):
        """
        Test injection operator when the sparse points are replicated over
        multiple MPI ranks.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        f.data[:] = 0.
        coords = [(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)]
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = 4.

        # Global view (left) and local view (right, after domain decomposition)
        # O is a grid point
        # x is a halo point
        # A, B, C, D are sparse points
        #                               Rank0           Rank1
        # O --- O --- O --- O           O --- O --- x   x --- O --- O
        # |  A  |     |     |           |  A  |     |   |     |     |
        # O --- O --- O --- O           O --- O --- x   x --- O --- O
        # |     |  C  |  B  |     -->   |     |  C  |   |  C  |  B  |
        # O --- O --- O --- O           x --- x --- x   x --- x --- x
        # |     |  D  |     |           Rank2           Rank3
        # O --- O --- O --- O           x --- x --- x   x --- x --- x
        #                               |     |  C  |   |  C  |  B  |
        #                               O --- O --- x   x --- O --- O
        #                               |     |  D  |   |  D  |     |
        #                               O --- O --- x   x --- O --- O
        #
        # Expected `f.data` (global view)
        #
        # 1.25 --- 1.25 --- 0.00 --- 0.00
        #  |        |        |        |
        # 1.25 --- 2.50 --- 2.50 --- 1.25
        #  |        |        |        |
        # 0.00 --- 2.50 --- 3.75 --- 1.25
        #  |        |        |        |
        # 0.00 --- 1.25 --- 1.25 --- 0.00

        op = Operator(sf.inject(field=f, expr=sf + 1))
        op.apply()

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:  # rank0
            assert np.all(f.data_ro_domain == [[1.25, 1.25], [1.25, 2.5]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:  # rank1
            assert np.all(f.data_ro_domain == [[0., 0.], [2.5, 1.25]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(f.data_ro_domain == [[0., 2.5], [0., 1.25]])
        elif RIGHT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(f.data_ro_domain == [[3.75, 1.25], [1.25, 0.]])

    @pytest.mark.parallel(mode=4)
    def test_interpolation_wodup(self):
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))

        f = Function(name='f', grid=grid, space_order=0)
        f.data[:] = 4.
        coords = [(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)]
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = 0.

        # This is the situation at this point
        # O is a grid point
        # * is a sparse point
        #
        # O --- O --- O --- O
        # |  *  |     |  *  |
        # O --- O --- O --- O
        # |     |     |     |
        # O --- O --- O --- O
        # |  *  |     |  *  |
        # O --- O --- O --- O

        op = Operator(sf.interpolate(expr=f))
        op.apply()

        assert np.all(sf.data == 4.)

    @pytest.mark.parallel(mode=4)
    def test_interpolation_dup(self):
        """
        Test interpolation operator when the sparse points are replicated over
        multiple MPI ranks.
        """
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
        x, y = grid.dimensions

        # Init Function+data
        f = Function(name='f', grid=grid)
        f.data[:] = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
        coords = np.array([(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)])
        sf = SparseFunction(name='sf', grid=grid, npoint=len(coords), coordinates=coords)
        sf.data[:] = 0.

        # Global view (left) and local view (right, after domain decomposition)
        # O is a grid point
        # x is a halo point
        # A, B, C, D are sparse points
        #                               Rank0           Rank1
        # O --- O --- O --- O           O --- O --- x   x --- O --- O
        # |  A  |     |     |           |  A  |     |   |     |     |
        # O --- O --- O --- O           O --- O --- x   x --- O --- O
        # |     |  C  |  B  |     -->   |     |  C  |   |  C  |  B  |
        # O --- O --- O --- O           x --- x --- x   x --- x --- x
        # |     |  D  |     |           Rank2           Rank3
        # O --- O --- O --- O           x --- x --- x   x --- x --- x
        #                               |     |  C  |   |  C  |  B  |
        #                               O --- O --- x   x --- O --- O
        #                               |     |  D  |   |  D  |     |
        #                               O --- O --- x   x --- O --- O
        #
        # The initial `f.data` is (global view)
        #
        # 1. --- 1. --- 1. --- 1.
        # |      |      |      |
        # 2. --- 2. --- 2. --- 2.
        # |      |      |      |
        # 3. --- 3. --- 3. --- 3.
        # |      |      |      |
        # 4. --- 4. --- 4. --- 4.
        #
        # Expected `sf.data` (global view)
        #
        # 1.5 --- 2.5 --- 2.5 --- 3.5

        op = Operator(sf.interpolate(expr=f))
        op.apply()

        assert np.all(sf.data == [1.5, 2.5, 2.5, 3.5][grid.distributor.myrank])

    @pytest.mark.parallel(mode=2)
    def test_subsampling(self):
        grid = Grid(shape=(40,))
        x = grid.dimensions[0]
        t = grid.stepping_dim
        time = grid.time_dim

        nt = 9

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        # Setup subsampled function
        factor = 4
        nsamples = (nt+factor-1)//factor
        times = ConditionalDimension('t_sub', parent=time, factor=factor)
        fsave = TimeFunction(name='fsave', grid=grid, save=nsamples, time_dim=times)

        eqns = [Eq(f.forward, f[t, x-1] + f[t, x+1]), Eq(fsave, f)]
        op = Operator(eqns)
        op.apply(time=nt-1)

        assert np.all(f.data_ro_domain[0] == fsave.data_ro_domain[nsamples-1])
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(fsave.data_ro_domain[nsamples-1, nt-1:] == 256.)
        else:
            assert np.all(fsave.data_ro_domain[nsamples-1, :-(nt-1)] == 256.)

        # Also check there are no redundant halo exchanges
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1
        # In particular, there is no need for a halo exchange within the conditional
        conditional = FindNodes(Conditional).visit(op)
        assert len(conditional) == 1
        assert len(FindNodes(Call).visit(conditional[0])) == 0

    @pytest.mark.parallel(mode=2)
    def test_arguments_subrange(self):
        """
        Test op.apply when a subrange is specified for a distributed dimension.
        """
        grid = Grid(shape=(16,))
        x = grid.dimensions[0]

        f = TimeFunction(name='f', grid=grid)

        op = Operator(Eq(f.forward, f + 1.))
        op.apply(time=0, x_m=4, x_M=11)

        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[1, :4] == 0.)
            assert np.all(f.data_ro_domain[1, 4:] == 1.)
        else:
            assert np.all(f.data_ro_domain[1, :-4] == 1.)
            assert np.all(f.data_ro_domain[1, -4:] == 0.)

    @pytest.mark.parallel(mode=2)
    def test_bcs_basic(self):
        """
        Test MPI in presence of boundary condition loops. Here, no halo exchange
        is expected (as there is no stencil in the computed expression) but we
        check that:

            * the left BC loop is computed by the leftmost rank only
            * the right BC loop is computed by the rightmost rank only
        """
        grid = Grid(shape=(20,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        thickness = 4

        u = TimeFunction(name='u', grid=grid, time_order=1)

        xleft = SubDimension.left(name='xleft', parent=x, thickness=thickness)
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)
        xright = SubDimension.right(name='xright', parent=x, thickness=thickness)

        t_in_centre = Eq(u[t+1, xi], 1)
        leftbc = Eq(u[t+1, xleft], u[t+1, xleft+1] + 1)
        rightbc = Eq(u[t+1, xright], u[t+1, xright-1] + 1)

        op = Operator([t_in_centre, leftbc, rightbc])

        op.apply(time_m=1, time_M=1)

        glb_pos_map = u.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(u.data_ro_domain[0, thickness:] == 1.)
            assert np.all(u.data_ro_domain[0, :thickness] == range(thickness+1, 1, -1))
        else:
            assert np.all(u.data_ro_domain[0, :-thickness] == 1.)
            assert np.all(u.data_ro_domain[0, -thickness:] == range(2, thickness+2))

    @pytest.mark.parallel(mode=2)
    def test_interior_w_stencil(self):
        grid = Grid(shape=(10,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u[t, x-1] + u[t, x+1] + 1, subdomain=grid.interior))
        op.apply(time=1)

        glb_pos_map = u.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(u.data_ro_domain[0, 1] == 2.)
            assert np.all(u.data_ro_domain[0, 2:] == 3.)
        else:
            assert np.all(u.data_ro_domain[0, -2] == 2.)
            assert np.all(u.data_ro_domain[0, :-2] == 3.)

    @pytest.mark.parallel(mode=4)
    def test_misc_dims(self):
        """
        Test MPI in presence of Functions with mixed distributed/replicated
        Dimensions, with only a strict subset of the Grid dimensions used.
        """
        dx = Dimension(name='dx')
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        glb_pos_map = grid.distributor.glb_pos_map
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2, save=4)
        c = Function(name='c', grid=grid, dimensions=(x, dx), shape=(4, 5))

        step = Eq(u.forward, (
            u[time, x-2, y] * c[x, 0]
            + u[time, x-1, y] * c[x, 1]
            + u[time, x, y] * c[x, 2]
            + u[time, x+1, y] * c[x, 3]
            + u[time, x+2, y] * c[x, 4]))

        for i in range(4):
            c.data[i, 0] = 1.0+i
            c.data[i, 1] = 1.0+i
            c.data[i, 2] = 3.0+i
            c.data[i, 3] = 6.0+i
            c.data[i, 4] = 5.0+i

        u.data[:] = 0.0
        u.data[0, 2, :] = 2.0

        op = Operator(step)

        op(time_m=0, time_M=0)

        if LEFT in glb_pos_map[x]:
            assert(np.all(u.data[1, 0, :] == 10.0))
            assert(np.all(u.data[1, 1, :] == 14.0))
        else:
            assert(np.all(u.data[1, 2, :] == 10.0))
            assert(np.all(u.data[1, 3, :] == 8.0))

    @pytest.mark.parallel(mode=9)
    def test_nontrivial_operator(self):
        """
        Test MPI in a non-trivial scenario: ::

            * 9 processes logically organised in a 3x3 cartesian grid (as opposed to
              most tests in this module, which only use 2 or 4 processed);
            * star-like stencil expression;
            * non-trivial Higdon-like BCs;
            * simultaneous presence of TimeFunction(grid), Function(grid), and
              Function(dimensions)
        """
        size_x, size_y = 9, 9
        tkn = 2

        # Grid and Dimensions
        grid = Grid(shape=(size_x, size_y,))
        x, y = grid.dimensions
        t = grid.stepping_dim

        # SubDimensions to implement BCs
        xl, yl = [SubDimension.left('%sl' % d.name, d, tkn) for d in [x, y]]
        xi, yi = [SubDimension.middle('%si' % d.name, d, tkn, tkn) for d in [x, y]]
        xr, yr = [SubDimension.right('%sr' % d.name, d, tkn) for d in [x, y]]

        # Functions
        u = TimeFunction(name='f', grid=grid)
        m = Function(name='m', grid=grid)
        c = Function(name='c', grid=grid, dimensions=(x,), shape=(size_x,))

        # Data initialization
        u.data_with_halo[:] = 0.
        m.data_with_halo[:] = 1.
        c.data_with_halo[:] = 0.

        # Equations
        c_init = Eq(c, 1.)
        eqn = Eq(u[t+1, xi, yi], u[t, xi, yi] + m[xi, yi] + c[xi] + 1.)
        bc_left = Eq(u[t+1, xl, yi], u[t+1, xl+1, yi] + 1.)
        bc_right = Eq(u[t+1, xr, yi], u[t+1, xr-1, yi] + 1.)
        bc_top = Eq(u[t+1, xi, yl], u[t+1, xi, yl+1] + 1.)
        bc_bottom = Eq(u[t+1, xi, yr], u[t+1, xi, yr-1] + 1.)

        op = Operator([c_init, eqn, bc_left, bc_right, bc_top, bc_bottom])
        op.apply(time=0)

        # Expected (global view):
        # 0 0 5 5 5 5 5 0 0
        # 0 0 4 4 4 4 4 0 0
        # 5 4 3 3 3 3 3 4 5
        # 5 4 3 3 3 3 3 4 5
        # 5 4 3 3 3 3 3 4 5
        # 5 4 3 3 3 3 3 4 5
        # 0 0 4 4 4 4 4 0 0
        # 0 0 5 5 5 5 5 0 0

        assert np.all(u.data_ro_domain[0] == 0)  # The write occures at t=1

        glb_pos_map = u.grid.distributor.glb_pos_map
        # Check cornes
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[0, 0, 5], [0, 0, 4], [5, 4, 3]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[5, 0, 0], [4, 0, 0], [3, 4, 5]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[5, 4, 3], [0, 0, 4], [0, 0, 5]])
        elif RIGHT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[3, 4, 5], [4, 0, 0], [5, 0, 0]])
        # Check sides
        if not glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[5, 4, 3], [5, 4, 3], [5, 4, 3]])
        elif not glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[3, 4, 5], [3, 4, 5], [3, 4, 5]])
        elif LEFT in glb_pos_map[x] and not glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[5, 5, 5], [4, 4, 4], [3, 3, 3]])
        elif RIGHT in glb_pos_map[x] and not glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == [[3, 3, 3], [4, 4, 4], [5, 5, 5]])
        # Check center
        if not glb_pos_map[x] and not glb_pos_map[y]:
            assert np.all(u.data_ro_domain[1] == 3)

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'overlap'), (4, 'full')])
    def test_coupled_eqs_mixed_dims(self):
        """
        Test an Operator that computes coupled equations over partly disjoint sets
        of Dimensions (e.g., one Eq over [x, y, z], the other Eq over [x, yi, zi]).
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        xi, yi = grid.interior.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=2)
        v = TimeFunction(name='v', grid=grid, space_order=2)

        u.data_with_halo[:] = 1.

        eqns = [Eq(u[t+1, x, y], u[t, x-1, y] + u[t, x, y] + u[t, x+1, y] + v + 1),
                Eq(v[t+1, x, yi],
                   (v[t, x, yi] + u[t, x, yi-1] + u[t, x, yi] + u[t, x, yi+1] + 1))]

        # `u`'s stencil is
        #
        #   *
        # * C *
        #   *
        #
        # Where C is a generic point (x, y) and * are the accessed neighbours

        op = Operator(eqns)
        op.apply(time=0)

        assert np.all(u.data_ro_domain[1] == 4.)

        glb_pos_map = v.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(v.data_ro_domain[1] == [[0, 4], [0, 4]])
        elif LEFT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(v.data_ro_domain[1] == [[4, 0], [4, 0]])
        elif RIGHT in glb_pos_map[x] and LEFT in glb_pos_map[y]:
            assert np.all(v.data_ro_domain[1] == [[0, 4], [0, 4]])
        elif RIGHT in glb_pos_map[x] and RIGHT in glb_pos_map[y]:
            assert np.all(v.data_ro_domain[1] == [[4, 0], [4, 0]])

        # Same checks as above, but exploiting the user API
        assert np.all(v.data_ro_domain[1, :, 0] == 0.)
        assert np.all(v.data_ro_domain[1, :, 1] == 4.)
        assert np.all(v.data_ro_domain[1, :, 2] == 4.)
        assert np.all(v.data_ro_domain[1, :, 3] == 0.)

    @pytest.mark.parallel(mode=2)
    def test_haloupdate_same_timestep(self):
        """
        Test an Operator that computes coupled equations in which the second
        one requires a halo update right after the computation of the first one.
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        u.data_with_halo[:] = 1.
        v = TimeFunction(name='v', grid=grid)
        v.data_with_halo[:] = 0.

        eqns = [Eq(u.forward, u + v + 1.),
                Eq(v.forward, u[t+1, x, y-1] + u[t+1, x, y] + u[t+1, x, y+1])]

        op = Operator(eqns)
        op.apply(time=0)

        assert np.all(v.data_ro_domain[-1, :, 1:-1] == 6.)

    @pytest.mark.parallel(mode=2)
    def test_haloupdate_same_timestep_v2(self):
        """
        Similar to test_haloupdate_same_timestep, but switching the expression that
        writes to subsequent time step. Also checks halo update call placement.
        MFE for issue #1483
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        u.data_with_halo[:] = 1.
        v = TimeFunction(name='v', grid=grid)
        v.data_with_halo[:] = 0.

        eqns = [Eq(u, u + v + 1.),
                Eq(v.forward, u[t, x, y-1] + u[t, x, y] + u[t, x, y+1])]

        op = Operator(eqns)

        titer = op.body[-1].body[0]
        assert titer.dim is grid.time_dim
        assert titer.nodes[0].body[0].body[0].is_List
        assert len(titer.nodes[0].body[0].body[0].body[0].body) == 1
        assert titer.nodes[0].body[0].body[0].body[0].body[0].is_Call

        op.apply(time=0)

        assert np.all(v.data_ro_domain[-1, :, 1:-1] == 6.)

    @pytest.mark.parallel(mode=4)
    def test_haloupdate_multi_op(self):
        """
        Test that halo updates are carried out correctly when multiple operators
        are applied consecutively.
        """
        a = np.arange(64).reshape((8, 8))
        grid = Grid(shape=a.shape, extent=(8, 8))

        so = 3
        dims = grid.dimensions
        f = Function(name='f', grid=grid, space_order=so)
        f.data[:] = a

        fo = Function(name='fo', grid=grid, space_order=so)

        for d in dims:
            rhs = generic_derivative(f, d, so, 1)
            expr = Eq(fo, rhs)
            op = Operator(expr)
            op.apply()
            f.data[:, :] = fo.data[:, :]

        assert (np.isclose(norm(f), 17.24904, atol=1e-4, rtol=0))

    @pytest.mark.parallel(mode=1)
    def test_haloupdate_issue_1613(self):
        """
        Test the HaloScheme construction and generation when using u.dt2.

        This stems from issue #1613.
        """
        configuration['mpi'] = True

        grid = Grid(shape=(10, 10))
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=4, time_order=2)

        eqns = [Eq(u.forward, u.dt2 + u.dx)]

        op = Operator(eqns)

        # The loc_indices must be along `t` (ie `t0`)
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 1
        dims = [i for i in calls[0].arguments if isinstance(i, Dimension)]
        assert len(dims) == 1
        assert dims[0].is_Modulo
        assert dims[0].origin is t

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag2'), (4, 'overlap2')])
    def test_cire(self):
        """
        Check correctness when the DSE extracts aliases and places them
        into offset-ed loop (nest). For example, the compiler may generate:

            for i = i_m - 1 to i_M + 1
              tmp[i] = f(a[i-1], a[i], a[i+1], ...)
            for i = i_m to i_M
              u[i] = g(tmp[i-1], tmp[i], ... a[i], ...)

        If the employed MPI scheme doesn't use comp/comm overlap (i.e., `basic`,
        `diag`), then it's not so different than most of the other tests seen in
        this module. However, with comp/comm overlap, which exploits the same loops
        to compute the boundary ("OWNED") regions, the situation is more delicate.
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u.data_with_halo[:] = 0.

        eqn = Eq(u.forward, ((u[t, x, y] + u[t, x+1, y+1])*3.*f +
                             (u[t, x+2, y+2] + u[t, x+3, y+3])*3.*f + 1))
        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('advanced', {'cire-mingain': 1}))

        op0(time_M=1)
        u0_norm = norm(u)

        u._data_with_inhalo[:] = 0.
        op1(time_M=1)
        u1_norm = norm(u)

        assert u0_norm == u1_norm

    @pytest.mark.parallel(mode=[(4, 'overlap2'), (4, 'diag2')])
    def test_cire_with_shifted_diagonal_halo_touch(self):
        """
        Like ``test_cire`` but now the diagonal halos required to compute
        the aliases are shifted due to the iteration space. Basically, this
        is checking that ``TimedAccess.touched_halo`` does the right thing
        using the information stored in ``.intervals``.
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u.data_with_halo[:] = 0.

        eqn = Eq(u.forward, ((u[t, x, y] + u[t, x+2, y])*3.*f +
                             (u[t, x+1, y+1] + u[t, x+3, y+1])*3.*f + 1))
        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('advanced', {'cire-mingain': 1}))

        op0(time_M=1)
        u0_norm = norm(u)

        u._data_with_inhalo[:] = 0.
        op1(time_M=1)
        u1_norm = norm(u)

        assert u0_norm == u1_norm

    @pytest.mark.parallel(mode=4)
    @pytest.mark.parametrize('opt_options', [
        {'cire-rotate': True},  # Issue #1490
        {'min-storage': True},  # Issue #1491
    ])
    def test_cire_options(self, opt_options):
        """
        MFEs for issues #1490 and #1491.
        """
        grid = Grid(shape=(128, 128, 128), dtype=np.float64)

        p = TimeFunction(name='p', grid=grid, time_order=2, space_order=8)
        p1 = TimeFunction(name='p', grid=grid, time_order=2, space_order=8)

        p.data[0, 40:80, 40:80, 40:80] = 0.12
        p1.data[0, 40:80, 40:80, 40:80] = 0.12

        eqn = Eq(p.forward, (p.dx).dx + (p.dy).dy + (p.dz).dz)

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('advanced', opt_options))

        # Check generated code
        arrays = [i for i in FindSymbols().visit(op1._func_table['bf0']) if i.is_Array]
        assert len(arrays) == 3
        assert 'haloupdate0' in op1._func_table
        # We expect exactly one halo exchange
        calls = FindNodes(Call).visit(op1)
        assert len(calls) == 5
        assert calls[0].name == 'haloupdate0'
        assert all(i.name == 'bf0' for i in calls[1:])

        op0.apply(time_M=1)
        op1.apply(time_M=1, p=p1)

        # TODO: we will tighten the tolerance, or switch to single precision,
        # or both, once issue #1438 is fixed
        assert np.allclose(p.data, p1.data, rtol=10e-11)

    @pytest.mark.parallel(mode=[(4, 'full')])
    def test_staggering(self):
        """
        Test MPI in presence of staggered grids.

        The equations are semantically meaningless, but they are designed to
        generate the kind of loop nest structure which is typical of *-elastic
        problems (e.g., visco-elastic).
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions

        so = 2
        ux = TimeFunction(name='ux', grid=grid, staggered=x, space_order=so)
        uxx = TimeFunction(name='uxx', grid=grid, staggered=NODE, space_order=so)
        uxy = TimeFunction(name='uxy', grid=grid, staggered=(x, y), space_order=so)

        eqns = [Eq(ux.forward, ux + 0.2*uxx.dx + uxy.dy + 0.5),
                Eq(uxx.forward, uxx + ux.forward.dx + ux.forward.dy + 1.),
                Eq(uxy.forward, 40.*uxy + ux.forward.dx + ux.forward.dy + 3.)]

        op = Operator(eqns)

        op(time_M=2)

        # Expected norms computed "manually" from sequential runs
        assert np.isclose(norm(ux), 6253.4349, rtol=1.e-4)
        assert np.isclose(norm(uxx), 80001.0304, rtol=1.e-4)
        assert np.isclose(norm(uxy), 61427.853, rtol=1.e-4)

    @pytest.mark.parallel(mode=2)
    def test_op_new_dist(self):
        """
        Test that an operator made with one distributor produces correct results
        when executed with a different distributor.
        """
        grid = Grid(shape=(10, 10), comm=MPI.COMM_SELF)
        grid2 = Grid(shape=(10, 10), comm=MPI.COMM_WORLD)

        u = TimeFunction(name='u', grid=grid, space_order=2)
        u2 = TimeFunction(name='u2', grid=grid2, space_order=2)

        x, y = np.ix_(np.linspace(-1, 1, grid.shape[0]),
                      np.linspace(-1, 1, grid.shape[1]))
        dx = x - 0.5
        dy = y
        u.data[0, :, :] = 1.0 * ((dx*dx + dy*dy) < 0.125)
        u2.data[0, :, :] = 1.0 * ((dx*dx + dy*dy) < 0.125)

        # Create some operator that requires MPI communication
        eqn = Eq(u.forward, u + 0.000001 * u.laplace)
        op = Operator(eqn)

        op.apply(u=u, time_M=10)
        op.apply(u=u2, time_M=10)

        assert abs(norm(u) - norm(u2)) < 1.e-3

    @pytest.mark.parallel(mode=[(4, 'full')])
    def test_misc_subdims(self):
        """
        Test MPI full mode with an Operator having:

            * A middle SubDimension in which at least one of the extremes has
              thickness 0;
            * A left SubDimension.

        Thus, only one of the two distributed Dimensions (x and y) induces
        a halo exchange

        Derived from issue https://github.com/devitocodes/devito/issues/1121
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        u.data_with_halo[:] = 1.

        xi = SubDimension.middle(name='xi', parent=x, thickness_left=0, thickness_right=1)
        yl = SubDimension.left(name='yl', parent=y, thickness=1)

        # A 5 point stencil expression
        eqn = Eq(u[t+1, xi, yl], (u[t, xi, yl] + u[t, xi-1, yl] + u[t, xi+1, yl] +
                                  u[t, xi, yl-1] + u[t, xi, yl+1]))

        op = Operator(eqn)

        # Halo exchanges metadata check-up
        msgs = [i for i in op.parameters if i.name.startswith('msg')]
        assert len(msgs) == 1
        msg = msgs.pop()
        assert len(msg.halos) == 2

        op(time_M=0)

        # Also try running it
        assert np.all(u.data[1, :-1, :1] == 5.)
        assert np.all(u.data[1, -1:] == 1.)
        assert np.all(u.data[1, :, 1:] == 1.)

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'full')])
    def test_misc_subdims_3D(self):
        """
        Test `SubDims` in 3D (so that spatial blocking is introduced).

        Derived from issue https://github.com/devitocodes/devito/issues/1309
        """
        grid = Grid(shape=(12, 12, 12))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        u.data_with_halo[:] = 1.

        xi = SubDimension.middle(name='xi', parent=x, thickness_left=2, thickness_right=2)
        yi = SubDimension.middle(name='yi', parent=y, thickness_left=2, thickness_right=2)
        zi = SubDimension.middle(name='zi', parent=z, thickness_left=2, thickness_right=2)

        # A 7 point stencil expression
        eqn = Eq(u[t+1, xi, yi, zi], (u[t, xi, yi, zi]
                                      + u[t, xi-1, yi, zi] + u[t, xi+1, yi, zi]
                                      + u[t, xi, yi-1, zi] + u[t, xi, yi+1, zi]
                                      + u[t, xi, yi, zi-1] + u[t, xi, yi, zi+1]))

        op = Operator(eqn)

        op(time_M=0)

        # Also try running it
        assert np.all(u.data[1, 2:-2, 2:-2, 2:-2] == 7.)
        assert np.all(u.data[1, 0:2, :, :] == 1.)
        assert np.all(u.data[1, -2:, :, :] == 1.)
        assert np.all(u.data[1, :, 0:2, :] == 1.)
        assert np.all(u.data[1, :, -2:, :] == 1.)
        assert np.all(u.data[1, :, :, 0:2] == 1.)
        assert np.all(u.data[1, :, :, -2:] == 1.)

    @pytest.mark.parallel(mode=[(4, 'full')])
    def test_custom_subdomain(self):
        """
        This test uses a custom SubDomain such that we end up with two loop
        nests with a data dependence across them inducing two halo exchanges,
        one for each loop nests. A crucial aspect of this test is that the data
        dependence is across a Dimension (xl) that does *not* require a halo
        exchange (xl is a local SubDimension). Unlike more typical cases of
        when a dependence occurs along time, here the dependence distance goes
        to infinity (there are indeed two consecutive separate loop nests), so
        a halo exchange *is* required even though the halo Dimension (y) is not
        the one inducing the dependence.
        """

        class mydomain(SubDomain):
            name = 'd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('left', 2), y: y}

        mydomain = mydomain()

        grid = Grid(shape=(8, 8), subdomains=mydomain)

        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name="u", grid=grid)
        v = TimeFunction(name="v", grid=grid)

        u.data[:] = 0.6
        v.data[:] = 0.4

        eqns = [Eq(u.forward, u[t, x, y-1] + 1, subdomain=grid.subdomains['d1']),
                Eq(v.forward, u[t+1, x+1, y+1] + 1, subdomain=grid.subdomains['d1'])]

        op = Operator(eqns, subs=grid.spacing_map)

        # We expect 2 halo-exchange sets of calls, for a total of 8 calls
        calls = FindNodes(Call).visit(op)
        assert len(calls) == 8

        # Compilation + run
        op(time_M=4)

        # Check numerical values
        assert np.isclose(norm(u), 23.70654, atol=1e-5, rtol=0)
        assert np.isclose(norm(v), 21.14994, atol=1e-5, rtol=0)

    @pytest.mark.parallel(mode=2)
    def test_overriding_from_different_grid(self):
        """
        MFE for issue #1629.
        """
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        xi = SubDimension.middle(name='xi', parent=x, thickness_left=3, thickness_right=3)
        yi = SubDimension.middle(name='yi', parent=y, thickness_left=3, thickness_right=3)
        u = TimeFunction(name='u', grid=grid, space_order=2, time_order=0)

        eqn = Eq(u.forward, u + 1).subs({x: xi, y: yi})
        op = Operator(eqn)

        grid2 = Grid(shape=(10, 10), dimensions=(x, y))
        u2 = TimeFunction(name='u', grid=grid2, space_order=2, time_order=0)

        op.apply(time_M=0, u=u2)
        assert np.all(u2.data[0, 3:-3, 3:-3] == 1.)

        grid3 = Grid(shape=(10, 10))
        u3 = TimeFunction(name='u', grid=grid3, space_order=2, time_order=0)

        op.apply(time_M=0, u=u3)
        assert np.all(u3.data[0, 3:-3, 3:-3] == 1.)


def gen_serial_norms(shape, so):
    """
    Computes the norms of the outputs in serial mode to compare with
    """
    day = np.datetime64('today')
    try:
        l = np.load("norms%s.npy" % len(shape), allow_pickle=True)
        assert l[-1] == day
    except:
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create solver from preset
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape],
                                tn=tn, space_order=so, nrec=nrec,
                                preset='layers-isotropic', dtype=np.float64)
        # Run forward operator
        rec, u, _ = solver.forward()
        Eu = norm(u)
        Erec = norm(rec)

        # Run adjoint operator
        srca, v, _ = solver.adjoint(rec=rec)
        Ev = norm(v)
        Esrca = norm(srca)

        np.save("norms%s.npy" % len(shape), (Eu, Erec, Ev, Esrca, day), allow_pickle=True)


class TestIsotropicAcoustic(object):

    """
    Test the isotropic acoustic wave equation with MPI.
    """
    _shapes = {1: (60,), 2: (60, 70), 3: (60, 70, 80)}
    _so = {1: 12, 2: 8, 3: 4}
    gen_serial_norms((60,), 12)
    gen_serial_norms((60, 70), 8)
    gen_serial_norms((60, 70, 80), 4)

    @cached_property
    def norms(self):
        return {1: np.load("norms1.npy", allow_pickle=True)[:-1],
                2: np.load("norms2.npy", allow_pickle=True)[:-1],
                3: np.load("norms3.npy", allow_pickle=True)[:-1]}

    @pytest.mark.parametrize('shape,kernel,space_order,save', [
        ((60, ), 'OT2', 4, False),
        ((60, 70), 'OT2', 8, False),
    ])
    @pytest.mark.parallel(mode=1)
    def test_adjoint_codegen(self, shape, kernel, space_order, save):
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape], kernel=kernel,
                                tn=500, space_order=space_order, nrec=130,
                                preset='layers-isotropic', dtype=np.float64)
        op_fwd = solver.op_fwd(save=save)
        fwd_calls = FindNodes(Call).visit(op_fwd)

        op_adj = solver.op_adj()
        adj_calls = FindNodes(Call).visit(op_adj)

        assert len(fwd_calls) == 1
        assert len(adj_calls) == 1

    def run_adjoint_F(self, nd):
        """
        Unlike `test_adjoint_F` in test_adjoint.py, here we explicitly check the norms
        of all Operator-evaluated Functions. The numbers we check against are derived
        "manually" from sequential runs of test_adjoint::test_adjoint_F
        """
        Eu, Erec, Ev, Esrca = self.norms[nd]
        shape = self._shapes[nd]
        so = self._so[nd]
        tn = 500.  # Final time
        nrec = 130  # Number of receivers

        # Create solver from preset
        solver = acoustic_setup(shape=shape, spacing=[15. for _ in shape],
                                tn=tn, space_order=so, nrec=nrec,
                                preset='layers-isotropic', dtype=np.float64)
        # Run forward operator
        rec, u, _ = solver.forward()

        assert np.isclose(norm(u) / Eu, 1.0)
        assert np.isclose(norm(rec) / Erec, 1.0)

        # Run adjoint operator
        srca, v, _ = solver.adjoint(rec=rec)

        assert np.isclose(norm(v) / Ev, 1.0)
        assert np.isclose(norm(srca) / Esrca, 1.0)

        # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
        term1 = inner(srca, solver.geometry.src)
        term2 = norm(rec)**2
        assert np.isclose((term1 - term2)/term1, 0., rtol=1.e-10)

    @pytest.mark.parametrize('nd', [1, 2, 3])
    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'diag'), (4, 'overlap'),
                                (4, 'overlap2'), (4, 'full')])
    def test_adjoint_F(self, nd):
        self.run_adjoint_F(nd)

    @pytest.mark.parallel(mode=[(8, 'diag2'), (8, 'full')])
    @switchconfig(openmp=False)
    def test_adjoint_F_no_omp(self):
        """
        ``run_adjoint_F`` with OpenMP disabled. By disabling OpenMP, we can
        practically scale up to higher process counts.
        """
        self.run_adjoint_F(3)


if __name__ == "__main__":
    configuration['mpi'] = True
    # TestDecomposition().test_reshape_left_right()
    # TestOperatorSimple().test_trivial_eq_2d()
    # TestOperatorSimple().test_num_comms('f[t,x-1,y] + f[t,x+1,y]', {'rc', 'lc'})
    # TestFunction().test_halo_exchange_bilateral()
    # TestSparseFunction().test_ownership(((1., 1.), (1., 3.), (3., 1.), (3., 3.)))
    # TestSparseFunction().test_local_indices([(0.5, 0.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5)], [[0.], [1.], [2.], [3.]])  # noqa
    # TestSparseFunction().test_scatter_gather()
    # TestOperatorAdvanced().test_nontrivial_operator()
    # TestOperatorAdvanced().test_interpolation_dup()
    # TestOperatorAdvanced().test_injection_wodup()
    TestIsotropicAcoustic().test_adjoint_F_no_omp()
