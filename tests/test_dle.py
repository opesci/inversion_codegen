from functools import reduce
from operator import mul

import numpy as np
import pytest

from devito import (Grid, Function, TimeFunction, SparseTimeFunction, SpaceDimension,
                    Dimension, SubDimension, Eq, Inc, Operator, info)
from devito.exceptions import InvalidArgument
from devito.ir.iet import Call, Iteration, Conditional, FindNodes, retrieve_iteration_tree
from devito.passes.iet.languages.openmp import OmpRegion
from devito.tools import as_tuple
from devito.types import Scalar


def get_blocksizes(op, opt, grid, blockshape, level=0):
    blocksizes = {'%s0_blk%d_size' % (d, level): v
                  for d, v in zip(grid.dimensions, blockshape)}
    blocksizes = {k: v for k, v in blocksizes.items() if k in op._known_arguments}
    # Sanity check
    if grid.dim == 1 or len(blockshape) == 0:
        assert len(blocksizes) == 0
        return {}
    try:
        if opt[1].get('blockinner'):
            assert len(blocksizes) >= 1
            if grid.dim == len(blockshape):
                assert len(blocksizes) == len(blockshape)
            else:
                assert len(blocksizes) <= len(blockshape)
        return blocksizes
    except AttributeError:
        assert len(blocksizes) == 0
        return {}


def _new_operator2(shape, time_order, blockshape=None, opt=None):
    blockshape = as_tuple(blockshape)
    grid = Grid(shape=shape, dtype=np.int32)
    infield = TimeFunction(name='infield', grid=grid, time_order=time_order)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = TimeFunction(name='outfield', grid=grid, time_order=time_order)

    stencil = Eq(outfield.forward.indexify(),
                 outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, opt=opt)

    blocksizes = get_blocksizes(op, opt, grid, blockshape)
    op(infield=infield, outfield=outfield, t=10, **blocksizes)

    return outfield, op


def _new_operator3(shape, blockshape0=None, blockshape1=None, opt=None):
    blockshape0 = as_tuple(blockshape0)
    blockshape1 = as_tuple(blockshape1)

    grid = Grid(shape=shape, extent=shape, dtype=np.float64)

    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=(2, 2, 2))
    u.data[0, :] = np.linspace(-1, 1, reduce(mul, shape)).reshape(shape)

    # Derive the stencil according to devito conventions
    op = Operator(Eq(u.forward, 0.5 * u.laplace + u), opt=opt)

    blocksizes0 = get_blocksizes(op, opt, grid, blockshape0, 0)
    blocksizes1 = get_blocksizes(op, opt, grid, blockshape1, 1)
    op.apply(u=u, t=10, **blocksizes0, **blocksizes1)

    return u.data[1, :], op


@pytest.mark.parametrize("shape", [(41,), (20, 33), (45, 31, 45)])
def test_composite_transformation(shape):
    wo_blocking, _ = _new_operator2(shape, time_order=2, opt='noop')
    w_blocking, _ = _new_operator2(shape, time_order=2, opt='advanced')

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("blockinner,exp_calls,exp_iters", [
    (False, 4, 5),
    (True, 8, 6)
])
def test_cache_blocking_structure(blockinner, exp_calls, exp_iters):
    # Check code structure
    _, op = _new_operator2((10, 31, 45), time_order=2,
                           opt=('blocking', {'blockinner': blockinner,
                                             'par-collapse-ncores': 1}))
    calls = FindNodes(Call).visit(op)
    assert len(calls) == exp_calls
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree) == exp_iters
    if blockinner:
        assert all(tree[i].dim.is_Incr for i in range(exp_iters))
    else:
        assert all(tree[i].dim.is_Incr for i in range(exp_iters-1))
        assert not tree[-1].dim.is_Incr

    # Check presence of openmp pragmas at the right place
    _, op = _new_operator2((10, 31, 45), time_order=2,
                           opt=('blocking', {'openmp': True,
                                             'blockinner': blockinner,
                                             'par-collapse-ncores': 1}))
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree.root.pragmas) == 1
    assert 'omp for' in tree.root.pragmas[0].value
    # Also, with omp parallelism enabled, the step increment must be != 0
    # to avoid omp segfaults at scheduling time (only certain omp implementations,
    # including Intel's)
    conditionals = FindNodes(Conditional).visit(op._func_table['bf0'].root)
    assert len(conditionals) == 1
    conds = conditionals[0].condition.args
    expected_guarded = tree[:2+blockinner]
    assert len(conds) == len(expected_guarded)
    assert all(i.lhs == j.step for i, j in zip(conds, expected_guarded))


def test_cache_blocking_structure_subdims():
    """
    Test that:

        * With local SubDimensions no-blocking is expected.
        * With non-local SubDimensions, blocking is expected.
    """
    grid = Grid(shape=(4, 4, 4))
    x, y, z = grid.dimensions
    xi, yi, zi = grid.interior.dimensions
    t = grid.stepping_dim
    xl = SubDimension.left(name='xl', parent=x, thickness=4)

    f = TimeFunction(name='f', grid=grid)

    assert xl.local

    # Local SubDimension -> no blocking expected
    op = Operator(Eq(f[t+1, xl, y, z], f[t, xl, y, z] + 1))
    assert len(op._func_table) == 0

    # Non-local SubDimension -> blocking expected
    op = Operator(Eq(f.forward, f + 1, subdomain=grid.interior))
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree) == 5
    assert tree[0].dim.is_Incr and tree[0].dim.parent is xi and tree[0].dim.root is x
    assert tree[1].dim.is_Incr and tree[1].dim.parent is yi and tree[1].dim.root is y
    assert tree[2].dim.is_Incr and tree[2].dim.parent is tree[0].dim and\
        tree[2].dim.root is x
    assert tree[3].dim.is_Incr and tree[3].dim.parent is tree[1].dim and\
        tree[3].dim.root is y
    assert not tree[4].dim.is_Incr and tree[4].dim is zi and tree[4].dim.parent is z


@pytest.mark.parallel(mode=[(1, 'full')])  # Shortcut to put loops in nested efuncs
def test_cache_blocking_structure_multiple_efuncs():
    """
    Test cache blocking in multiple nested elemental functions.
    """
    grid = Grid(shape=(4, 4, 4))
    x, y, z = grid.dimensions

    u = TimeFunction(name="u", grid=grid, space_order=2)
    U = TimeFunction(name="U", grid=grid, space_order=2)
    src = SparseTimeFunction(name="src", grid=grid, nt=3, npoint=1,
                             coordinates=np.array([(0.5, 0.5, 0.5)]))

    eqns = [Eq(u.forward, u.dx)]
    eqns += src.inject(field=u.forward, expr=src)
    eqns += [Eq(U.forward, U.dx + u.forward)]

    op = Operator(eqns)

    for i in ['bf0', 'bf1']:
        assert i in op._func_table
        iters = FindNodes(Iteration).visit(op._func_table[i].root)
        assert len(iters) == 5
        assert iters[0].dim.parent is x
        assert iters[1].dim.parent is y
        assert iters[4].dim is z
        assert iters[2].dim.parent is iters[0].dim
        assert iters[3].dim.parent is iters[1].dim


@pytest.mark.parametrize("shape", [(10,), (10, 45), (20, 33), (10, 31, 45), (45, 31, 45)])
@pytest.mark.parametrize("time_order", [2])
@pytest.mark.parametrize("blockshape", [2, (3, 3), (9, 20), (2, 9, 11), (7, 15, 23)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_time_loop(shape, time_order, blockshape, blockinner):
    wo_blocking, _ = _new_operator2(shape, time_order, opt='noop')
    w_blocking, _ = _new_operator2(shape, time_order, blockshape,
                                   opt=('blocking', {'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((25, 25, 46), (25, 25, 46)),
    ((25, 25, 46), (7, 25, 46)),
    ((25, 25, 46), (25, 25, 7)),
    ((25, 25, 46), (25, 7, 46)),
    ((25, 25, 46), (5, 25, 7)),
    ((25, 25, 46), (10, 3, 46)),
    ((25, 25, 46), (25, 7, 11)),
    ((25, 25, 46), (8, 2, 4)),
    ((25, 25, 46), (2, 4, 8)),
    ((25, 25, 46), (4, 8, 2)),
    ((25, 46), (25, 7)),
    ((25, 46), (7, 46))
])
def test_cache_blocking_edge_cases(shape, blockshape):
    time_order = 2
    wo_blocking, _ = _new_operator2(shape, time_order, opt='noop')
    w_blocking, _ = _new_operator2(shape, time_order, blockshape,
                                   opt=('blocking', {'blockinner': True}))
    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((3, 3), (3, 3)),
    ((4, 4), (3, 4)),
    ((5, 5), (3, 4)),
    ((6, 6), (3, 4)),
    ((7, 7), (3, 4)),
    ((8, 8), (3, 4)),
    ((9, 9), (3, 4)),
    ((10, 10), (3, 4)),
    ((11, 11), (3, 4)),
    ((12, 12), (3, 4)),
    ((13, 13), (3, 4)),
    ((14, 14), (3, 4)),
    ((15, 15), (3, 4))
])
def test_cache_blocking_edge_cases_highorder(shape, blockshape):
    wo_blocking, a = _new_operator3(shape, opt='noop')
    w_blocking, b = _new_operator3(shape, blockshape, opt=('blocking',
                                                           {'blockinner': True}))

    assert np.allclose(wo_blocking, w_blocking, rtol=1e-12)


@pytest.mark.parametrize("blockshape0,blockshape1,exception", [
    ((24, 24, 40), (24, 24, 40), False),
    ((24, 24, 40), (4, 4, 4), False),
    ((24, 24, 40), (8, 8, 8), False),
    ((20, 20, 12), (4, 4, 4), False),
    ((28, 32, 16), (14, 16, 8), False),
    ((12, 12, 60), (4, 12, 4), False),
    ((12, 12, 60), (4, 5, 4), True),  # not a perfect divisor
    ((12, 12, 60), (24, 4, 4), True),  # bigger than outer block
])
def test_cache_blocking_hierarchical(blockshape0, blockshape1, exception):
    shape = (51, 102, 71)

    wo_blocking, a = _new_operator3(shape, opt='noop')
    try:
        w_blocking, b = _new_operator3(shape, blockshape0, blockshape1,
                                       opt=('blocking', {'blockinner': True,
                                                         'blocklevels': 2}))
        assert not exception
        assert np.allclose(wo_blocking, w_blocking, rtol=1e-12)
    except InvalidArgument:
        assert exception
    except:
        assert False


@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_imperfect_nest(blockinner):
    """
    Test that a non-perfect Iteration nest is blocked correctly.
    """
    grid = Grid(shape=(4, 4, 4), dtype=np.float64)

    u = TimeFunction(name='u', grid=grid, space_order=2)
    v = TimeFunction(name='v', grid=grid, space_order=2)

    eqns = [Eq(u.forward, v.laplace),
            Eq(v.forward, u.forward.dz)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt=('advanced', {'blockinner': blockinner}))

    # First, check the generated code
    trees = retrieve_iteration_tree(op1._func_table['bf0'].root)
    assert len(trees) == 2
    assert len(trees[0]) == len(trees[1])
    assert all(i is j for i, j in zip(trees[0][:4], trees[1][:4]))
    assert trees[0][4] is not trees[1][4]
    assert trees[0].root.dim.is_Incr
    assert trees[1].root.dim.is_Incr
    assert op1.parameters[7] is trees[0][0].step
    assert op1.parameters[10] is trees[0][1].step

    u.data[:] = 0.2
    v.data[:] = 1.5
    op0(time_M=0)

    u1 = TimeFunction(name='u1', grid=grid, space_order=2)
    v1 = TimeFunction(name='v1', grid=grid, space_order=2)

    u1.data[:] = 0.2
    v1.data[:] = 1.5
    op1(u=u1, v=v1, time_M=0)

    assert np.all(u.data == u1.data)
    assert np.all(v.data == v1.data)


@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_imperfect_nest_v2(blockinner):
    """
    Test that a non-perfect Iteration nest is blocked correctly. This
    is slightly different than ``test_cache_blocking_imperfect_nest``
    as here only one Iteration gets blocked.
    """
    shape = (16, 16, 16)
    grid = Grid(shape=shape, dtype=np.float64)

    u = TimeFunction(name='u', grid=grid, space_order=4)
    u.data[:] = np.linspace(0, 1, reduce(mul, shape), dtype=np.float64).reshape(shape)

    eq = Eq(u.forward, 0.01*u.dy.dy)

    op0 = Operator(eq, opt='noop')
    op1 = Operator(eq, opt=('cire-sops', {'blockinner': blockinner}))
    op2 = Operator(eq, opt=('advanced-fsg', {'blockinner': blockinner}))

    # First, check the generated code
    trees = retrieve_iteration_tree(op2._func_table['bf0'].root)
    assert len(trees) == 2
    assert len(trees[0]) == len(trees[1])
    assert all(i is j for i, j in zip(trees[0][:2], trees[1][:2]))
    assert trees[0][2] is not trees[1][2]
    assert trees[0].root.dim.is_Incr
    assert trees[1].root.dim.is_Incr
    assert op2.parameters[6] is trees[0].root.step

    op0(time_M=0)

    u1 = TimeFunction(name='u1', grid=grid, space_order=4)
    u1.data[:] = np.linspace(0, 1, reduce(mul, shape), dtype=np.float64).reshape(shape)

    op1(time_M=0, u=u1)

    u2 = TimeFunction(name='u2', grid=grid, space_order=4)
    u2.data[:] = np.linspace(0, 1, reduce(mul, shape), dtype=np.float64).reshape(shape)

    op2(time_M=0, u=u2)

    assert np.allclose(u.data, u1.data, rtol=1e-07)
    assert np.allclose(u.data, u2.data, rtol=1e-07)


class TestNodeParallelism(object):

    def test_nthreads_generation(self):
        grid = Grid(shape=(10, 10))

        f = TimeFunction(name='f', grid=grid)

        eq = Eq(f.forward, f + 1)

        op0 = Operator(eq, openmp=True)

        # `nthreads` must appear among the Operator parameters
        assert op0.nthreads in op0.parameters

        # `nthreads` is bindable to a runtime value
        assert op0.nthreads._arg_values()

    @pytest.mark.parametrize('exprs,expected', [
        # trivial 1D
        (['Eq(fa[x], fa[x] + fb[x])'],
         (True,)),
        # trivial 1D
        (['Eq(t0, fa[x] + fb[x])', 'Eq(fa[x], t0 + 1)'],
         (True,)),
        # trivial 2D
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x,y], t0 + 1)'],
         (True, False)),
        # outermost parallel, innermost sequential
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x,y+1], t0 + 1)'],
         (True, False)),
        # outermost sequential, innermost parallel
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x+1,y], t0 + 1)'],
         (False, True)),
        # outermost sequential, innermost parallel
        (['Eq(fc[x,y], fc[x+1,y+1] + fc[x-1,y])'],
         (False, True)),
        # outermost parallel w/ repeated dimensions (hence irregular dependencies)
        # both `x` and `y` are parallel-if-atomic loops
        (['Inc(t0, fc[x,x] + fd[x,y+1])', 'Eq(fc[x,x], t0 + 1)'],
         (True, False)),
        # outermost sequential, innermost sequential (classic skewing example)
        (['Eq(fc[x,y], fc[x,y+1] + fc[x-1,y])'],
         (False, False)),
        # skewing-like over two Eqs
        (['Eq(t0, fc[x,y+2] + fc[x-1,y+2])', 'Eq(fc[x,y+1], t0 + 1)'],
         (False, False)),
        # outermost parallel, innermost sequential w/ double tensor write
        (['Eq(fc[x,y], fc[x,y+1] + fd[x-1,y])', 'Eq(fd[x-1,y+1], fd[x-1,y] + fc[x,y+1])'],
         (True, False, False)),
        # outermost sequential, innermost parallel w/ mixed dimensions
        (['Eq(fc[x+1,y], fc[x,y+1] + fc[x,y])', 'Eq(fc[x+1,y], 2. + fc[x,y+1])'],
         (False, True)),
    ])
    def test_iterations_ompized(self, exprs, expected):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions  # noqa

        fa = Function(name='fa', grid=grid, dimensions=(x,), shape=(4,))  # noqa
        fb = Function(name='fb', grid=grid, dimensions=(x,), shape=(4,))  # noqa
        fc = Function(name='fc', grid=grid)  # noqa
        fd = Function(name='fd', grid=grid)  # noqa
        t0 = Scalar(name='t0')  # noqa

        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        op = Operator(eqns, opt='openmp')

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == len(expected)

        # Check for presence of pragma omp
        for i, j in zip(iterations, expected):
            pragmas = i.pragmas
            if j is True:
                assert len(pragmas) == 1
                pragma = pragmas[0]
                assert 'omp for' in pragma.value
            else:
                for k in pragmas:
                    assert 'omp for' not in k.value

    def test_dynamic_nthreads(self):
        grid = Grid(shape=(16, 16, 16))
        f = TimeFunction(name='f', grid=grid)
        sf = SparseTimeFunction(name='sf', grid=grid, npoint=1, nt=5)

        eqns = [Eq(f.forward, f + 1)]
        eqns += sf.interpolate(f)

        op = Operator(eqns, opt='openmp')

        parregions = FindNodes(OmpRegion).visit(op)
        assert len(parregions) == 2

        # Check suitable `num_threads` appear in the generated code
        # Not very elegant, but it does the trick
        assert 'num_threads(nthreads)' in str(parregions[0].header[0])
        assert 'num_threads(nthreads_nonaffine)' in str(parregions[1].header[0])

        # Check `op` accepts the `nthreads*` kwargs
        op.apply(time=0)
        op.apply(time_m=1, time_M=1, nthreads=4)
        op.apply(time_m=1, time_M=1, nthreads=4, nthreads_nonaffine=2)
        op.apply(time_m=1, time_M=1, nthreads_nonaffine=2)
        assert np.all(f.data[0] == 2.)

        # Check the actual value assumed by `nthreads` and `nthreads_nonaffine`
        assert op.arguments(time=0, nthreads=123)['nthreads'] == 123
        assert op.arguments(time=0, nthreads_nonaffine=100)['nthreads_nonaffine'] == 100

    @pytest.mark.parametrize('eqns,expected,blocking', [
        ('[Eq(f, 2*f)]', [2, 0, 0], False),
        ('[Eq(u, 2*u)]', [0, 2, 0, 0], False),
        ('[Eq(u, 2*u)]', [3, 0, 0, 0, 0, 0], True),
        ('[Eq(u, 2*u), Eq(f, u.dzr)]', [0, 2, 0, 0, 0], False)
    ])
    def test_collapsing(self, eqns, expected, blocking):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)  # noqa
        u = TimeFunction(name='u', grid=grid)  # noqa

        eqns = eval(eqns)

        if blocking:
            op = Operator(eqns, opt=('blocking', 'simd', 'openmp',
                                     {'blockinner': True, 'par-collapse-ncores': 1,
                                      'par-collapse-work': 0}))
            iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        else:
            op = Operator(eqns, opt=('simd', 'openmp', {'par-collapse-ncores': 1,
                                                        'par-collapse-work': 0}))
            iterations = FindNodes(Iteration).visit(op)

        assert len(iterations) == len(expected)

        # Check for presence of pragma omp + collapse clause
        for i, j in zip(iterations, expected):
            if j > 0:
                assert len(i.pragmas) == 1
                pragma = i.pragmas[0]
                assert 'omp for collapse(%d)' % j in pragma.value
            else:
                for k in i.pragmas:
                    assert 'omp for collapse' not in k.value

    def test_collapsing_v2(self):
        """
        MFE from issue #1478.
        """
        n = 8
        m = 8
        nx, ny, nchi, ncho = 12, 12, 1, 1
        x, y = SpaceDimension("x"), SpaceDimension("y")
        ci, co = Dimension("ci"), Dimension("co")
        i, j = Dimension("i"), Dimension("j")
        grid = Grid((nx, ny), dtype=np.float32, dimensions=(x, y))

        X = Function(name="xin", dimensions=(ci, x, y),
                     shape=(nchi, nx, ny), grid=grid, space_order=n//2)
        dy = Function(name="dy", dimensions=(co, x, y),
                      shape=(ncho, nx, ny), grid=grid, space_order=n//2)
        dW = Function(name="dW", dimensions=(co, ci, i, j), shape=(ncho, nchi, n, m),
                      grid=grid)

        eq = [Eq(dW[co, ci, i, j],
                 dW[co, ci, i, j] + dy[co, x, y]*X[ci, x+i-n//2, y+j-m//2])
              for i in range(n) for j in range(m)]

        op = Operator(eq, opt=('advanced', {'openmp': True}))

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == 4
        assert iterations[0].ncollapsed == 1
        assert iterations[1].is_Vectorized
        assert iterations[2].is_Sequential
        assert iterations[3].is_Sequential

    def test_scheduling(self):
        """
        Affine iterations -> #pragma omp ... schedule(dynamic,1) ...
        Non-affine iterations -> #pragma omp ... schedule(dynamic,chunk_size) ...
        """
        grid = Grid(shape=(11, 11))

        u = TimeFunction(name='u', grid=grid, time_order=2, save=5, space_order=0)
        sf1 = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=5)

        eqns = [Eq(u.forward, u + 1)]
        eqns += sf1.interpolate(u)

        op = Operator(eqns, opt=('openmp', {'par-dynamic-work': 0}))

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == 4
        assert iterations[1].is_Affine
        assert 'schedule(dynamic,1)' in iterations[1].pragmas[0].value
        assert not iterations[3].is_Affine
        assert 'schedule(dynamic,chunk_size)' in iterations[3].pragmas[0].value

    @pytest.mark.parametrize('so', [0, 1, 2])
    @pytest.mark.parametrize('dim', [0, 1, 2])
    def test_array_reduction(self, so, dim):
        """
        Test generation of OpenMP reduction clauses involving Function's.
        """
        grid = Grid(shape=(3, 3, 3))
        d = grid.dimensions[dim]

        f = Function(name='f', shape=(3,), dimensions=(d,), grid=grid, space_order=so)
        u = TimeFunction(name='u', grid=grid)

        op = Operator(Inc(f, u + 1), opt=('openmp', {'par-collapse-ncores': 1}))

        iterations = FindNodes(Iteration).visit(op)
        assert "reduction(+:f[0:f_vec->size[0]])" in iterations[1].pragmas[0].value

        try:
            op(time_M=1)
        except:
            # Older gcc <6.1 don't support reductions on array
            info("Un-supported older gcc version for array reduction")
            assert True
            return

        assert np.allclose(f.data, 18)

    def test_incs_no_atomic(self):
        """
        Test that `Inc`'s don't get a `#pragma omp atomic` if performing
        an increment along a fully parallel loop.
        """
        grid = Grid(shape=(8, 8, 8))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        # Format: u(t, x, nastyness) += 1
        uf = u[t, x, f, z]

        # All loops get collapsed, but the `y` and `z` loops are PARALLEL_IF_ATOMIC,
        # hence an atomic pragma is expected
        op0 = Operator(Inc(uf, 1), opt=('advanced', {'openmp': True,
                                                     'par-collapse-ncores': 1}))
        assert 'collapse(3)' in str(op0)
        assert 'atomic' in str(op0)

        # Now only `x` is parallelized
        op1 = Operator([Eq(v[t, x, 0, 0], v[t, x, 0, 0] + 1), Inc(uf, 1)],
                       opt=('advanced', {'openmp': True, 'par-collapse-ncores': 1}))
        assert 'collapse(1)' in str(op1)
        assert 'atomic' not in str(op1)


class TestNestedParallelism(object):

    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1),
                      opt=('blocking', 'openmp', {'par-nested': 0,
                                                  'par-collapse-ncores': 10000,
                                                  'par-dynamic-work': 0}))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction

        # Does it produce the right result
        op.apply(t_M=9)
        assert np.all(u.data[0] == 10)

        # Try again but this time supplying specific values for the num_threads
        u.data[:] = 0.
        op.apply(t_M=9, nthreads=1, nthreads_nested=2)
        assert np.all(u.data[0] == 10)
        assert op.arguments(t_M=9, nthreads_nested=2)['nthreads_nested'] == 2

        iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        assert iterations[0].pragmas[0].value == 'omp for collapse(1) schedule(dynamic,1)'
        assert iterations[2].pragmas[0].value == ('omp parallel for collapse(1) '
                                                  'schedule(dynamic,1) '
                                                  'num_threads(nthreads_nested)')

    def test_collapsing(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1),
                      opt=('blocking', 'openmp', {'par-nested': 0,
                                                  'par-collapse-ncores': 1,
                                                  'par-collapse-work': 0,
                                                  'par-dynamic-work': 0}))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction

        # Does it produce the right result
        op.apply(t_M=9)
        assert np.all(u.data[0] == 10)

        iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        assert iterations[0].pragmas[0].value == 'omp for collapse(2) schedule(dynamic,1)'
        assert iterations[2].pragmas[0].value == ('omp parallel for collapse(2) '
                                                  'schedule(dynamic,1) '
                                                  'num_threads(nthreads_nested)')

    def test_multiple_subnests_v0(self):
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=3)

        eqn = Eq(u.forward, ((u[t, x, y, z] + u[t, x+1, y+1, z+1])*3.*f +
                             (u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3.*f + 1))
        op = Operator(eqn, opt=('advanced', {'openmp': True,
                                             'cire-mingain': 1,
                                             'par-nested': 0,
                                             'par-collapse-ncores': 1,
                                             'par-dynamic-work': 0}))

        trees = retrieve_iteration_tree(op._func_table['bf0'].root)
        assert len(trees) == 2

        assert trees[0][0] is trees[1][0]
        assert trees[0][0].pragmas[0].value ==\
            'omp for collapse(2) schedule(dynamic,1)'
        assert trees[0][2].pragmas[0].value == ('omp parallel for collapse(2) '
                                                'schedule(dynamic,1) '
                                                'num_threads(nthreads_nested)')
        assert trees[1][2].pragmas[0].value == ('omp parallel for collapse(2) '
                                                'schedule(dynamic,1) '
                                                'num_threads(nthreads_nested)')

    def test_multiple_subnests_v1(self):
        """
        Unlike ``test_multiple_subnestes_v0``, now we use the ``cire-rotate=True``
        option, which trades some of the inner parallelism for a smaller working set.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=3)

        eqn = Eq(u.forward, ((u[t, x, y, z] + u[t, x+1, y+1, z+1])*3.*f +
                             (u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3.*f + 1))
        op = Operator(eqn, opt=('advanced', {'openmp': True,
                                             'cire-mingain': 1,
                                             'cire-rotate': True,
                                             'par-nested': 0,
                                             'par-collapse-ncores': 1,
                                             'par-dynamic-work': 0}))

        trees = retrieve_iteration_tree(op._func_table['bf0'].root)
        assert len(trees) == 2

        assert trees[0][0] is trees[1][0]
        assert trees[0][0].pragmas[0].value ==\
            'omp for collapse(2) schedule(dynamic,1)'
        assert not trees[0][2].pragmas
        assert not trees[0][3].pragmas
        assert trees[0][4].pragmas[0].value == ('omp parallel for collapse(1) '
                                                'schedule(dynamic,1) '
                                                'num_threads(nthreads_nested)')
        assert not trees[1][2].pragmas
        assert trees[1][3].pragmas[0].value == ('omp parallel for collapse(1) '
                                                'schedule(dynamic,1) '
                                                'num_threads(nthreads_nested)')

    @pytest.mark.parametrize('blocklevels', [1, 2])
    def test_nested_cache_blocking_structure_subdims(self, blocklevels):
        """
        Test that:

            * With non-local SubDimensions, nested blocking works fine when expected.
            * With non-local SubDimensions, hierarchical nested blocking works fine
            when expected.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        xi, yi, zi = grid.interior.dimensions
        xl = SubDimension.left(name='xl', parent=x, thickness=4)

        f = TimeFunction(name='f', grid=grid)

        assert xl.local

        # Non-local SubDimension -> nested blocking can works as expected
        op = Operator(Eq(f.forward, f + 1, subdomain=grid.interior),
                      opt=('blocking', 'openmp',
                           {'par-nested': 0, 'blocklevels': blocklevels,
                            'par-collapse-ncores': 2,
                            'par-dynamic-work': 0}))
        trees = retrieve_iteration_tree(op._func_table['bf0'].root)
        assert len(trees) == 1
        tree = trees[0]
        assert len(tree) == 5 + (blocklevels - 1) * 2
        assert tree[0].dim.is_Incr and tree[0].dim.parent is xi and tree[0].dim.root is x
        assert tree[1].dim.is_Incr and tree[1].dim.parent is yi and tree[1].dim.root is y
        assert tree[2].dim.is_Incr and tree[2].dim.parent is tree[0].dim and\
            tree[2].dim.root is x
        assert tree[3].dim.is_Incr and tree[3].dim.parent is tree[1].dim and\
            tree[3].dim.root is y

        if blocklevels == 1:
            assert not tree[4].dim.is_Incr and tree[4].dim is zi and\
                tree[4].dim.parent is z
        elif blocklevels == 2:
            assert tree[3].dim.is_Incr and tree[3].dim.parent is tree[1].dim and\
                tree[3].dim.root is y
            assert tree[4].dim.is_Incr and tree[4].dim.parent is tree[2].dim and\
                tree[4].dim.root is x
            assert tree[5].dim.is_Incr and tree[5].dim.parent is tree[3].dim and\
                tree[5].dim.root is y
            assert not tree[6].dim.is_Incr and tree[6].dim is zi and\
                tree[6].dim.parent is z

        assert trees[0][0].pragmas[0].value ==\
            'omp for collapse(2) schedule(dynamic,1)'
        assert trees[0][2].pragmas[0].value == ('omp parallel for collapse(2) '
                                                'schedule(dynamic,1) '
                                                'num_threads(nthreads_nested)')
