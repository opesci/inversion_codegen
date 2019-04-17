from sympy import sin  # noqa
import numpy as np
import pytest

from conftest import skipif, EVAL, x, y, z  # noqa
from devito import (Eq, Inc, Constant, Function, TimeFunction, SparseTimeFunction,  # noqa
                    SubDimension, Grid, Operator, switchconfig, configuration)
from devito.ir import Stencil, FlowGraph, FindSymbols, retrieve_iteration_tree  # noqa
from devito.dle import BlockDimension
from devito.dse import common_subexprs_elimination, collect
from devito.symbolics import (xreplace_constrained, iq_timeinvariant, iq_timevarying,
                              estimate_cost, pow_to_mul)
from devito.tools import generator
from devito.types import Scalar

from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, AcquisitionGeometry
from examples.seismic.tti import AnisotropicWaveSolver

pytestmark = skipif(['yask', 'ops'])

# Tests for DLE will be needed


# Acoustic

def run_acoustic_forward(dse=None):
    shape = (50, 50, 50)
    spacing = (10., 10., 10.)
    nbpml = 10
    nrec = 101
    t0 = 0.0
    tn = 250.0

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=3., vp_bottom=4.5,
                       spacing=spacing, shape=shape, nbpml=nbpml)

    # Source and receiver geometries
    src_coordinates = np.empty((1, len(spacing)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_coordinates[:, 1:] = src_coordinates[0, 1:]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=t0, tn=tn, src_type='Ricker', f0=0.010)

    solver = AcousticWaveSolver(model, geometry, dse=dse, dle='noop')
    rec, u, _ = solver.forward(save=False)

    return u, rec


def test_acoustic_rewrite_basic():
    ret1 = run_acoustic_forward(dse=None)
    ret2 = run_acoustic_forward(dse='basic')

    assert np.allclose(ret1[0].data, ret2[0].data, atol=10e-5)
    assert np.allclose(ret1[1].data, ret2[1].data, atol=10e-5)


# TTI

def tti_operator(dse=False, dle='advanced', space_order=4):
    nrec = 101
    t0 = 0.0
    tn = 250.
    nbpml = 10
    shape = (50, 50, 50)
    spacing = (20., 20., 20.)

    # Two layer model for true velocity
    model = demo_model('layers-tti', ratio=3, nbpml=nbpml, space_order=space_order,
                       shape=shape, spacing=spacing)

    # Source and receiver geometries
    src_coordinates = np.empty((1, len(spacing)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_coordinates[:, 1:] = src_coordinates[0, 1:]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=t0, tn=tn, src_type='Gabor', f0=0.010)

    return AnisotropicWaveSolver(model, geometry, space_order=space_order, dse=dse)


@pytest.fixture(scope="session")
def tti_nodse():
    operator = tti_operator(dse=None)
    rec, u, v, _ = operator.forward()
    return v, rec


def test_tti_rewrite_basic(tti_nodse):
    operator = tti_operator(dse='basic')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-3)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-3)


def test_tti_rewrite_advanced(tti_nodse):
    operator = tti_operator(dse='advanced')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-1)


def test_tti_rewrite_aggressive(tti_nodse):
    operator = tti_operator(dse='aggressive')
    rec, u, v, _ = operator.forward(kernel='centered', save=False)

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-1)

    # Also check that DLE's loop blocking with DSE=aggressive does the right thing
    # There should be exactly two BlockDimensions; bugs in the past were generating
    # either code with no blocking (zero BlockDimensions) or code with four
    # BlockDimensions (i.e., Iteration folding was somewhat broken)
    op = operator.op_fwd(kernel='centered', save=False)
    block_dims = [i for i in op.dimensions if isinstance(i, BlockDimension)]
    assert len(block_dims) == 2

    # Also, in this operator, we expect six temporary Arrays:
    # * four Arrays are allocated on the heap
    # * two Arrays are allocated on the stack and only appear within an efunc
    arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
    assert len(arrays) == 3
    assert all(i._mem_heap and not i._mem_external for i in arrays)
    arrays = [i for i in FindSymbols().visit(op._func_table['bf0'].root) if i.is_Array]
    assert len(arrays) == 5
    assert all(not i._mem_external for i in arrays)
    assert len([i for i in arrays if i._mem_heap]) == 3
    assert len([i for i in arrays if i._mem_stack]) == 2


@skipif(['nompi'])
@pytest.mark.parallel(mode=[(1, 'full')])
def test_tti_rewrite_aggressive_wmpi():
    tti_nodse = tti_operator(dse=None)
    rec0, u0, v0, _ = tti_nodse.forward(kernel='centered', save=False)
    tti_agg = tti_operator(dse='aggressive')
    rec1, u1, v1, _ = tti_agg.forward(kernel='centered', save=False)

    assert np.allclose(v0.data, v1.data, atol=10e-1)
    assert np.allclose(rec0.data, rec1.data, atol=10e-1)


@switchconfig(profiling='advanced')
@pytest.mark.parametrize('kernel,space_order,expected', [
    ('centered', 8, 148), ('centered', 16, 264)
])
def test_tti_rewrite_aggressive_opcounts(kernel, space_order, expected):
    operator = tti_operator(dse='aggressive', space_order=space_order)
    _, _, _, summary = operator.forward(kernel=kernel, save=False)
    assert summary['section1'].ops == expected


# DSE manipulation


def test_scheduling_after_rewrite():
    """Tests loop scheduling after DSE-induced expression hoisting."""
    grid = Grid((10, 10))
    u1 = TimeFunction(name="u1", grid=grid, save=10, time_order=2)
    u2 = TimeFunction(name="u2", grid=grid, time_order=2)
    sf1 = SparseTimeFunction(name='sf1', grid=grid, npoint=1, nt=10)
    const = Function(name="const", grid=grid, space_order=2)

    # Deliberately inject into u1, rather than u1.forward, to create a WAR
    eqn1 = Eq(u1.forward, u1 + sin(const))
    eqn2 = sf1.inject(u1.forward, expr=sf1)
    eqn3 = Eq(u2.forward, u2 - u1.dt2 + sin(const))

    op = Operator([eqn1] + eqn2 + [eqn3])
    trees = retrieve_iteration_tree(op)

    # Check loop nest structure
    assert len(trees) == 4
    assert all(i.dim == j for i, j in zip(trees[0], grid.dimensions))  # time invariant
    assert trees[1][0].dim == trees[2][0].dim == trees[3][0].dim == grid.time_dim


@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(tu, ti0 + ti1 + 5.)'],
     ['ti0[x, y, z] + ti1[x, y, z]']),
    # more ops
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(t0, 0.2)', 'Eq(t1, t0 + 2.)',
      'Eq(tw, 2. + ti0*t1)', 'Eq(tu, (ti0*ti1*t0) + (ti1*tv) + (t1 + ti1)*tw)'],
     ['t1*ti0[x, y, z]', 't1 + ti1[x, y, z]', 't0*ti0[x, y, z]*ti1[x, y, z]']),
    # wrapped
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(t0, 0.2)', 'Eq(t1, t0 + 2.)', 'Eq(tv, 2.4)',
      'Eq(tu, ((ti0*ti1*t0)*tv + (ti0*ti1*tv)*t1))'],
     ['t0*ti0[x, y, z]*ti1[x, y, z]', 't1*ti0[x, y, z]*ti1[x, y, z]']),
])
def test_xreplace_constrained_time_invariants(tu, tv, tw, ti0, ti1, t0, t1,
                                              exprs, expected):
    exprs = EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1)
    counter = generator()
    make = lambda: Scalar(name='r%d' % counter()).indexify()
    processed, found = xreplace_constrained(exprs, make,
                                            iq_timeinvariant(FlowGraph(exprs)),
                                            lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(ti0, 3.)', 'Eq(tv, 2.4)', 'Eq(tu, tv + 5. + ti0)'],
     ['tv[t, x, y, z] + 5.0']),
    # more ops
    (['Eq(tv, 2.4)', 'Eq(tw, tv*2.3)', 'Eq(ti1, 4.)', 'Eq(ti0, 3. + ti1)',
      'Eq(tu, tv*tw*4.*ti0 + ti1*tv)'],
     ['4.0*tv[t, x, y, z]*tw[t, x, y, z]']),
    # wrapped
    (['Eq(tv, 2.4)', 'Eq(tw, tv*tw*2.3)', 'Eq(ti1, 4.)', 'Eq(ti0, 3. + ti1)',
      'Eq(tu, ((tv + 4.)*ti0*ti1 + (tv + tw)/3.)*ti1*t0)'],
     ['tv[t, x, y, z] + 4.0',
      '0.333333333333333*tv[t, x, y, z] + 0.333333333333333*tw[t, x, y, z]']),
])
def test_xreplace_constrained_time_varying(tu, tv, tw, ti0, ti1, t0, t1,
                                           exprs, expected):
    exprs = EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1)
    counter = generator()
    make = lambda: Scalar(name='r%d' % counter()).indexify()
    processed, found = xreplace_constrained(exprs, make,
                                            iq_timevarying(FlowGraph(exprs)),
                                            lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(tu, (tv + tw + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     ['ti0[x, y, z] + ti1[x, y, z]',
      'r0*(t0 + t1) + r0*(tv[t, x, y, z] + tw[t, x, y, z] + 5.0)']),
    # across expressions
    (['Eq(tu, tv*4 + tw*5 + tw*5*t0)', 'Eq(tv, tw*5)'],
     ['5*tw[t, x, y, z]', 'r0 + 5*t0*tw[t, x, y, z] + 4*tv[t, x, y, z]', 'r0']),
    # intersecting
    pytest.param(['Eq(tu, ti0*ti1 + ti0*ti1*t0 + ti0*ti1*t0*t1)'],
                 ['ti0*ti1', 'r0', 'r0*t0', 'r0*t0*t1'],
                 marks=pytest.mark.xfail),
])
def test_common_subexprs_elimination(tu, tv, tw, ti0, ti1, t0, t1, exprs, expected):
    counter = generator()
    make = lambda: Scalar(name='r%d' % counter()).indexify()
    processed = common_subexprs_elimination(EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1),
                                            make)
    assert len(processed) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(processed, expected))


@pytest.mark.parametrize('expr,expected', [
    ('2*fa[x] + fb[x]', '2*fa[x] + fb[x]'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('fa[x]**2 + fb[x]**3', 'fa[x]*fa[x] + fb[x]*fb[x]*fb[x]'),
    ('3*fa[x]**4', '3*(fa[x]*fa[x]*fa[x]*fa[x])'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('1/(fa[x]**2)', '1/(fa[x]*fa[x])'),
    ('1/(fa[x] + fb[x])', '1/(fa[x] + fb[x])'),
    ('3*sin(fa[x])**2', '3*(sin(fa[x])*sin(fa[x]))'),
])
def test_pow_to_mul(fa, fb, expr, expected):
    assert str(pow_to_mul(eval(expr))) == expected


@pytest.mark.parametrize('exprs,expected', [
    # none (different distance)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x])'],
     {'fa[x] + fb[x]': 'None', 'fa[x+1] + fb[x]': 'None'}),
    # none (different dimension)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] + fb[y])'],
     {'fa[x] + fb[x]': 'None', 'fa[x] + fb[y]': 'None'}),
    # none (different operation)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] - fb[x])'],
     {'fa[x] + fb[x]': 'None', 'fa[x] - fb[x]': 'None'}),
    # simple
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x+1])', 'Eq(t2, fa[x-1] + fb[x-1])'],
     {'fa[x] + fb[x]': 'Stencil([(x, {-1, 0, 1})])'}),
    # 2D simple
    (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(t1, fc[x+1,y+1] + fd[x+1,y+1])'],
     {'fc[x,y] + fd[x,y]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # 2D with stride
    (['Eq(t0, fc[x,y] + fd[x+1,y+2])', 'Eq(t1, fc[x+1,y+1] + fd[x+2,y+3])'],
     {'fc[x,y] + fd[x+1,y+2]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # 2D with subdimensions
    (['Eq(t0, fc[xi,yi] + fd[xi+1,yi+2])', 'Eq(t1, fc[xi+1,yi+1] + fd[xi+2,yi+3])'],
     {'fc[xi,yi] + fd[xi+1,yi+2]': 'Stencil([(xi, {0, 1}), (yi, {0, 1})])'}),
    # 2D with constant access
    (['Eq(t0, fc[x,y]*fc[x,0] + fd[x,y])', 'Eq(t1, fc[x+1,y+1]*fc[x+1,0] + fd[x+1,y+1])'],
     {'fc[x,y]*fc[x,0] + fd[x,y]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # 2D with different shapes
    (['Eq(t0, fc[x,y]*fa[x] + fd[x,y])', 'Eq(t1, fc[x+1,y+1]*fa[x+1] + fd[x+1,y+1])'],
     {'fc[x,y]*fa[x] + fd[x,y]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # complex (two 2D aliases with stride inducing relaxation)
    (['Eq(t0, fc[x,y] + fd[x+1,y+2])', 'Eq(t1, fc[x+1,y+1] + fd[x+2,y+3])',
      'Eq(t2, fc[x-2,y-2]*3. + fd[x+2,y+2])', 'Eq(t3, fc[x-4,y-4]*3. + fd[x,y])'],
     {'fc[x,y] + fd[x+1,y+2]': 'Stencil([(x, {-1, 0, 1}), (y, {-1, 0, 1})])',
      '3.*fc[x-3,y-3] + fd[x+1,y+1]': 'Stencil([(x, {-1, 0, 1}), (y, {-1, 0, 1})])'}),
])
def test_collect_aliases(fc, fd, exprs, expected):
    grid = Grid(shape=(4, 4))
    x, y = grid.dimensions  # noqa
    xi, yi = grid.interior.dimensions  # noqa

    t0 = Scalar(name='t0')  # noqa
    t1 = Scalar(name='t1')  # noqa
    t2 = Scalar(name='t2')  # noqa
    t3 = Scalar(name='t3')  # noqa
    fa = Function(name='fa', grid=grid, shape=(4,), dimensions=(x,))  # noqa
    fb = Function(name='fb', grid=grid, shape=(4,), dimensions=(x,))  # noqa
    fc = Function(name='fc', grid=grid)  # noqa
    fd = Function(name='fd', grid=grid)  # noqa

    # List/dict comprehension would need explicit locals/globals mappings to eval
    for i, e in enumerate(list(exprs)):
        exprs[i] = eval(e)
    for k, v in list(expected.items()):
        expected[eval(k)] = eval(v)

    aliases = collect(exprs)

    assert len(aliases) > 0

    for k, v in aliases.items():
        assert ((len(v.aliased) == 1 and expected[k] is None) or
                v.anti_stencil == expected[k])


@pytest.mark.parametrize('expr,expected', [
    ('Eq(t0, t1)', 0),
    ('Eq(t0, fa[x] + fb[x])', 1),
    ('Eq(t0, fa[x + 1] + fb[x - 1])', 1),
    ('Eq(t0, fa[fb[x+1]] + fa[x])', 1),
    ('Eq(t0, fa[fb[x+1]] + fc[x+2, y+1])', 1),
    ('Eq(t0, t1*t2)', 1),
    ('Eq(t0, 2.*t0*t1*t2)', 3),
    ('Eq(t0, cos(t1*t2))', 2),
    ('Eq(t0, 2.*t0*t1*t2 + t0*fa[x+1])', 5),
    ('Eq(t0, (2.*t0*t1*t2 + t0*fa[x+1])*3. - t0)', 7),
    ('[Eq(t0, (2.*t0*t1*t2 + t0*fa[x+1])*3. - t0), Eq(t0, cos(t1*t2))]', 9),
])
def test_estimate_cost(fa, fb, fc, t0, t1, t2, expr, expected):
    # Note: integer arithmetic isn't counted
    assert estimate_cost(EVAL(expr, fa, fb, fc, t0, t1, t2)) == expected


@pytest.mark.parametrize('exprs,exp_u,exp_v', [
    (['Eq(s, 0, implicit_dims=(x, y))', 'Eq(s, s + 4, implicit_dims=(x, y))',
      'Eq(u, s)'], 4, 0),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Eq(s, s + s + 4, implicit_dims=(x, y))',
      'Eq(s, s + 4, implicit_dims=(x, y))', 'Eq(u, s)'], 8, 0),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Inc(s, 4, implicit_dims=(x, y))',
      'Eq(u, s)'], 4, 0),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Inc(s, 4, implicit_dims=(x, y))', 'Eq(v, s)',
      'Eq(u, s)'], 4, 4),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Inc(s, 4, implicit_dims=(x, y))', 'Eq(v, s)',
      'Eq(s, s + 4, implicit_dims=(x, y))', 'Eq(u, s)'], 8, 4),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Inc(s, 4, implicit_dims=(x, y))', 'Eq(v, s)',
      'Inc(s, 4, implicit_dims=(x, y))', 'Eq(u, s)'], 8, 4),
    (['Eq(u, 0)', 'Inc(u, 4)', 'Eq(v, u)', 'Inc(u, 4)'], 8, 4),
    (['Eq(u, 1)', 'Eq(v, 4)', 'Inc(u, v)', 'Inc(v, u)'], 5, 9),
])
def test_makeit_ssa(exprs, exp_u, exp_v):
    """
    A test building Operators with non-trivial sequences of input expressions
    that push hard on the `makeit_ssa` utility function.
    """
    grid = Grid(shape=(4, 4))
    x, y = grid.dimensions  # noqa
    u = Function(name='u', grid=grid)  # noqa
    v = Function(name='v', grid=grid)  # noqa
    s = Scalar(name='s')  # noqa

    # List comprehension would need explicit locals/globals mappings to eval
    for i, e in enumerate(list(exprs)):
        exprs[i] = eval(e)

    op = Operator(exprs)
    op.apply()

    assert np.all(u.data == exp_u)
    assert np.all(v.data == exp_v)


@pytest.mark.parametrize('dse', ['noop', 'basic', 'advanced', 'aggressive'])
@pytest.mark.parametrize('dle', ['noop', 'advanced', 'speculative'])
def test_time_dependent_split(dse, dle):
    grid = Grid(shape=(10, 10))
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2, save=3)
    v = TimeFunction(name='v', grid=grid, time_order=2, space_order=0, save=3)

    # The second equation needs a full loop over x/y for u then
    # a full one over x.y for v
    eq = [Eq(u.forward, 2 + grid.time_dim),
          Eq(v.forward, u.forward.dx + u.forward.dy + 1)]
    op = Operator(eq, dse=dse, dle=dle)

    trees = retrieve_iteration_tree(op)
    assert len(trees) == 2

    op()

    assert np.allclose(u.data[2, :, :], 3.0)
    assert np.allclose(v.data[1, 1:-1, 1:-1], 1.0)
