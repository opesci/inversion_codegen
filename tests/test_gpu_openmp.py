import numpy as np
import pytest
from sympy import cos

from conftest import skipif
from devito import (Grid, Dimension, Function, TimeFunction, Eq, Inc, solve,
                    Operator, switchconfig, norm)
from devito.exceptions import InvalidOperator
from devito.ir.iet import Block, FindNodes, retrieve_iteration_tree
from devito.mpi.routines import IrecvCall, IsendCall
from examples.seismic import TimeAxis, RickerSource, Receiver


class TestCodeGeneration(object):

    @switchconfig(platform='nvidiaX')
    def test_init_omp_env(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u.dx+1))

        assert str(op.body[0].body[0]) == ('if (deviceid != -1)\n'
                                           '{\n  omp_set_default_device(deviceid);\n}')

    @pytest.mark.parallel(mode=1)
    @switchconfig(platform='nvidiaX')
    def test_init_omp_env_w_mpi(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u.dx+1), opt=('advanced', {'gpu-direct': True}))

        assert str(op.body[0].body[0]) ==\
            ('if (deviceid != -1)\n'
             '{\n  omp_set_default_device(deviceid);\n}\n'
             'else\n'
             '{\n  int rank = 0;\n'
             '  MPI_Comm_rank(comm,&rank);\n'
             '  int ngpus = omp_get_num_devices();\n'
             '  omp_set_default_device((rank)%(ngpus));\n}')

    @switchconfig(platform='nvidiaX')
    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), language='openmp')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert op.body[2].header[0].value ==\
            ('omp target enter data map(to: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert str(op.body[2].footer[0]) == ''
        assert op.body[2].footer[1].contents[0].value ==\
            ('omp target update from(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[2].footer[1].contents[1].value ==\
            ('omp target exit data map(release: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]]) if(devicerm)')

        # Currently, advanced-fsg mode == advanced mode
        op1 = Operator(Eq(u.forward, u + 1), language='openmp', opt='advanced-fsg')
        assert str(op) == str(op1)

    @switchconfig(platform='nvidiaX')
    def test_basic_customop(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), language='openmp', opt='openmp')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'

        try:
            Operator(Eq(u.forward, u + 1), language='openmp', opt='openacc')
        except InvalidOperator:
            assert True
        except:
            assert False

    @switchconfig(platform='nvidiaX')
    def test_multiple_eqns(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        op = Operator([Eq(u.forward, u + v + 1), Eq(v.forward, u + v + 4)])

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        for i, f in enumerate([u, v]):
            assert op.body[2].header[i].value ==\
                ('omp target enter data map(to: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[2].footer[i+1].contents[0].value ==\
                ('omp target update from(%(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[2].footer[i+1].contents[1].value ==\
                ('omp target exit data map(release: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]]) '
                 'if(devicerm)' % {'n': f.name})

    @switchconfig(platform='nvidiaX')
    def test_multiple_loops(self):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=2)
        v = TimeFunction(name='v', grid=grid, space_order=2)

        eqns = [Eq(f, g*2),
                Eq(u.forward, u + v*f),
                Eq(v.forward, u.forward.dx + v*f + 4)]

        op = Operator(eqns, opt='noop')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 3

        # All loop nests must have been parallelized
        assert trees[0][0].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert trees[1][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert trees[2][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'

        # Check `u` and `v`
        for i, f in enumerate([u, v], 1):
            assert op.body[2].header[i].value ==\
                ('omp target enter data map(to: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[2].footer[i+1].contents[0].value ==\
                ('omp target update from(%(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[2].footer[i+1].contents[1].value ==\
                ('omp target exit data map(release: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]]) '
                 'if(devicerm)' % {'n': f.name})

        # Check `f`
        assert op.body[2].header[0].value ==\
            ('omp target enter data map(to: f[0:f_vec->size[0]]'
             '[0:f_vec->size[1]][0:f_vec->size[2]])')
        assert op.body[2].footer[1].contents[0].value ==\
            ('omp target update from(f[0:f_vec->size[0]]'
             '[0:f_vec->size[1]][0:f_vec->size[2]])')
        assert op.body[2].footer[1].contents[1].value ==\
            ('omp target exit data map(release: f[0:f_vec->size[0]]'
             '[0:f_vec->size[1]][0:f_vec->size[2]]) if(devicerm)')

        # Check `g` -- note that unlike `f`, this one should be `delete` upon
        # exit, not `from`
        assert op.body[2].header[3].value ==\
            ('omp target enter data map(to: g[0:g_vec->size[0]]'
             '[0:g_vec->size[1]][0:g_vec->size[2]])')
        assert op.body[2].footer[4].value ==\
            ('omp target exit data map(delete: g[0:g_vec->size[0]]'
             '[0:g_vec->size[1]][0:g_vec->size[2]])'
             ' if(devicerm && (g_vec->size[0] != 0) && (g_vec->size[1] != 0)'
             ' && (g_vec->size[2] != 0))')

    @switchconfig(platform='nvidiaX')
    def test_array_rw(self):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=2)

        eqn = Eq(u.forward, u*cos(f*2))

        op = Operator(eqn)

        assert len(op.body[2].header) == 7
        assert str(op.body[2].header[0]) == 'float (*r0)[y_size][z_size];'
        assert op.body[2].header[1].text ==\
            'posix_memalign((void**)&r0, 64, sizeof(float[x_size][y_size][z_size]))'
        assert op.body[2].header[2].value ==\
            ('omp target enter data map(alloc: r0[0:x_size][0:y_size][0:z_size])'
             '')

        assert len(op.body[2].footer) == 6
        assert str(op.body[2].footer[0]) == ''
        assert op.body[2].footer[1].value ==\
            ('omp target exit data map(delete: r0[0:x_size][0:y_size][0:z_size])'
             ' if((x_size != 0) && (y_size != 0) && (z_size != 0))')
        assert op.body[2].footer[2].text == 'free(r0)'

    @switchconfig(platform='nvidiaX')
    def test_function_wo(self):
        grid = Grid(shape=(3, 3, 3))
        i = Dimension(name='i')

        f = Function(name='f', shape=(1,), dimensions=(i,), grid=grid)
        u = TimeFunction(name='u', grid=grid)

        eqns = [Eq(u.forward, u + 1),
                Eq(f[0], u[0, 0, 0, 0])]

        op = Operator(eqns, opt='noop')

        assert len(op.body[2].header) == 2
        assert len(op.body[2].footer) == 2
        assert op.body[2].header[0].value ==\
            ('omp target enter data map(to: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert str(op.body[2].header[1]) == ''
        assert str(op.body[2].footer[0]) == ''
        assert op.body[2].footer[1].contents[0].value ==\
            ('omp target update from(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[2].footer[1].contents[1].value ==\
            ('omp target exit data map(release: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]]) if(devicerm)')

    @switchconfig(platform='nvidiaX')
    def test_timeparallel_reduction(self):
        grid = Grid(shape=(3, 3, 3))
        i = Dimension(name='i')

        f = Function(name='f', shape=(1,), dimensions=(i,), grid=grid)
        u = TimeFunction(name='u', grid=grid)

        op = Operator(Inc(f[0], u + 1), opt='noop')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1
        tree = trees[0]
        assert tree.root.is_Sequential
        assert all(i.is_ParallelRelaxed and not i.is_Parallel for i in tree[1:])

        # The time loop is not in OpenMP canonical form, so it won't be parallelized
        assert not tree.root.pragmas
        assert len(tree[1].pragmas) == 1
        assert tree[1].pragmas[0].value ==\
            ('omp target teams distribute parallel for collapse(3)'
             ' reduction(+:f[0])')

    @pytest.mark.parallel(mode=1)
    @switchconfig(platform='nvidiaX')
    def test_gpu_direct(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u.dx+1), opt=('advanced', {'gpu-direct': True}))

        for f, v in op._func_table.items():
            for node in FindNodes(Block).visit(v.root):
                if type(node.children[0][0]) in (IrecvCall, IsendCall):
                    assert node.header[0].value ==\
                        ('omp target data use_device_ptr(%s)' %
                         node.children[0][0].arguments[0].name)


class TestOperator(object):

    @skipif('nodevice')
    def test_op_apply(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid, dtype=np.int32)

        op = Operator(Eq(u.forward, u + 1))

        # Make sure we've indeed generated OpenMP offloading code
        assert 'omp target' in str(op)

        time_steps = 1000
        op.apply(time_M=time_steps)

        assert np.all(np.array(u.data[0, :, :, :]) == time_steps)

    def iso_acoustic(self, opt):
        shape = (101, 101)
        extent = (1000, 1000)
        origin = (0., 0.)

        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        grid = Grid(shape=shape, extent=extent, origin=origin)

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

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
        m = Function(name='m', grid=grid)
        m.data[:] = 1./(v*v)

        pde = m * u.dt2 - u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward))

        src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
        rec_term = rec.interpolate(expr=u.forward)

        op = Operator([stencil] + src_term + rec_term, opt=opt)

        # Make sure we've indeed generated OpenMP offloading code
        assert 'omp target' in str(op)

        op(time=time_range.num-1, dt=dt)

        assert np.isclose(norm(rec), 490.55, atol=1e-2, rtol=0)

    @skipif('nodevice')
    @pytest.mark.parametrize('blocklevels', [0, 1])
    def test_iso_acoustic(self, blocklevels):
        TestOperator().iso_acoustic(opt=('advanced', {'blocklevels': blocklevels}))

    @pytest.mark.parallel(mode=[2, 4])
    @skipif('nodevice')
    def test_gpu_direct(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid, dtype=np.int32)

        op = Operator(Eq(u.forward, u + 1), opt=('advanced', {'gpu-direct': True}))

        # Make sure we've indeed generated OpenMP offloading code
        assert 'omp target' in str(op)

        time_steps = 1000
        op.apply(time_M=time_steps)

        assert np.all(np.array(u.data[0, :, :, :]) == time_steps)

    @pytest.mark.parallel(mode=[2, 4])
    @skipif('nodevice')
    @pytest.mark.parametrize('blocklevels', [0, 1])
    def test_mpi_iso_acoustic(self, blocklevels):
        TestOperator().iso_acoustic(opt=('advanced', {'blocklevels': blocklevels}))
