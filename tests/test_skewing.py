import pytest

from sympy import Min, Max

from devito import Grid, Dimension, Eq, Function, TimeFunction, Operator # noqa
from devito.ir import Expression, Iteration, FindNodes
from devito.symbolics import INT


class TestCodeGenSkewing(object):

    '''
    Test code generation with blocking+skewing, tests adapted from test_operator.py
    '''
    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],u[t0,x-time+1,y-time+1,z+1]+1)']),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)']),
    ])
    def test_skewed_bounds(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', {'skewing': True}))
        op.apply(time_M=5)
        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 1

        iters = FindNodes(Iteration).visit(op)
        assert len(iters) == 6
        assert iters[1].dim.parent is x
        assert iters[2].dim.parent is y
        assert iters[5].dim is z
        assert iters[3].dim.parent is iters[1].dim
        assert iters[4].dim.parent is iters[2].dim

        assert (iters[3].symbolic_min == (iters[1].dim + time))
        assert (iters[3].symbolic_max == INT(Min(iters[1].dim + time +
                                                 iters[1].dim.symbolic_incr - 1,
                                                 Max(iters[1].dim.symbolic_max,
                                                 iters[1].dim.symbolic_max + time))))
        assert (iters[4].symbolic_min == (iters[2].dim + time))
        assert (iters[4].symbolic_max == INT(Min(iters[2].dim + time +
                                                 iters[2].dim.symbolic_incr - 1,
                                                 Max(iters[2].dim.symbolic_max,
                                                 iters[2].dim.symbolic_max + time))))

        assert (iters[5].symbolic_min == (iters[5].dim.symbolic_min))
        assert (iters[5].symbolic_max == (iters[5].dim.symbolic_max))
        skewed = [i.expr for i in FindNodes(Expression).visit(op)]
        assert str(skewed[0]).replace(' ', '') == expected

    '''
    Test code generation with skewing, tests adapted from test_operator.py
    '''
    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u, u + 1)',
          'Eq(u[x+1,y+1,z+1],u[x+1,y+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[x+1,y+1,z+1],v[x+1,y+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[x+1,y+1,z+1],v[x+1,y+1,z+1]+1)']),
    ])
    def test_no_sequential(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions

        u = Function(name='u', grid=grid)  # noqa
        v = Function(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', {'skewing': True}))
        op.apply()
        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 0

        iters = FindNodes(Iteration).visit(op)

        assert len(iters) == 3
        assert iters[0].dim is x
        assert iters[1].dim is y
        assert iters[2].dim is z

        skewed = [i.expr for i in FindNodes(Expression).visit(op)]
        assert str(skewed[0]).replace(' ', '') == expected

    '''
    Test code generation with skewing only
    '''
    @pytest.mark.parametrize('expr, expected, skewing, blockinner', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],u[t0,x-time+1,y-time+1,z+1]+1)', True, False]),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)', True, False]),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)', True, False]),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x+1,y+1,z+1],v[t0,x+1,y+1,z+1]+1)', False, False]),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z-time+1],v[t0,x-time+1,y-time+1,z-time+1]+1)',
          True, True]),
    ])
    def test_skewing_codegen(self, expr, expected, skewing, blockinner):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', {'blocklevels': 0, 'skewing': skewing,
                                             'blockinner': blockinner}))
        op.apply(time_M=5)

        iters = FindNodes(Iteration).visit(op)

        assert len(iters) == 4
        assert iters[0].dim is time
        assert iters[1].dim is x
        assert iters[2].dim is y
        assert iters[3].dim is z

        skewed = [i.expr for i in FindNodes(Expression).visit(op)]

        if skewing and not blockinner:
            assert (iters[1].symbolic_min == (iters[1].dim.symbolic_min + time))
            assert (iters[1].symbolic_max == (iters[1].dim.symbolic_max + time))
            assert (iters[2].symbolic_min == (iters[2].dim.symbolic_min + time))
            assert (iters[2].symbolic_max == (iters[2].dim.symbolic_max + time))
            assert (iters[3].symbolic_min == (iters[3].dim.symbolic_min))
            assert (iters[3].symbolic_max == (iters[3].dim.symbolic_max))
        elif skewing and blockinner:
            assert (iters[1].symbolic_min == (iters[1].dim.symbolic_min + time))
            assert (iters[1].symbolic_max == (iters[1].dim.symbolic_max + time))
            assert (iters[2].symbolic_min == (iters[2].dim.symbolic_min + time))
            assert (iters[2].symbolic_max == (iters[2].dim.symbolic_max + time))
            assert (iters[3].symbolic_min == (iters[3].dim.symbolic_min + time))
            assert (iters[3].symbolic_max == (iters[3].dim.symbolic_max + time))
        elif not skewing and not blockinner:
            assert (iters[1].symbolic_min == (iters[1].dim.symbolic_min))
            assert (iters[1].symbolic_max == (iters[1].dim.symbolic_max))
            assert (iters[2].symbolic_min == (iters[2].dim.symbolic_min))
            assert (iters[2].symbolic_max == (iters[2].dim.symbolic_max))
            assert (iters[3].symbolic_min == (iters[3].dim.symbolic_min))
            assert (iters[3].symbolic_max == (iters[3].dim.symbolic_max))

        assert str(skewed[0]).replace(' ', '') == expected
