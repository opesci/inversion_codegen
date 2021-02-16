import weakref

import numpy as np
import pytest

from devito import (Grid, Function, TimeFunction, SparseFunction, SparseTimeFunction,
                    ConditionalDimension, SubDimension, Constant, Operator, Eq, Dimension,
                    DefaultDimension, _SymbolCache, clear_cache, solve, VectorFunction,
                    TensorFunction, TensorTimeFunction, VectorTimeFunction)
from devito.types import Scalar, Symbol, NThreadsBase, DeviceID, NPThreads, ThreadID


@pytest.fixture
def operate_on_empty_cache():
    """
    To be used by tests that assert against the cache size. There are two
    reasons this is required:

        * Most symbolic objects embed further symbolic objects. For example,
          Function embeds Dimension, DerivedDimension embed a parent Dimension,
          and so on. The embedded objects require more than one call to
          `clear_cache` to be evicted (typically two -- the first call
          evicts the main object, then the children become unreferenced and so
          they are evicted upon the second call). So, depending on what tests
          were executed before, it is possible that one `clear_cache()` evicts
          more than expected, making it impossible to assert against cache sizes.
        * Due to some global symbols in `conftest.py`, it is possible that when
          for example a SparseFunction is instantiated, fewer symbolic object than
          expected are created, since some of them are available from the cache
          already.
    """
    old_cache = _SymbolCache.copy()
    _SymbolCache.clear()
    yield
    _SymbolCache.update(old_cache)


class TestHashing(object):

    """
    Test hashing of symbolic objects.
    """

    def test_constant(self):
        """Test that different Constants have different hash value."""
        c0 = Constant(name='c')
        c1 = Constant(name='c')
        assert c0 is not c1
        assert hash(c0) != hash(c1)

    def test_dimension(self):
        """Test that different Dimensions have different hash value."""
        d0 = Dimension(name='d')
        s0 = Scalar(name='s')
        d1 = Dimension(name='d', spacing=s0)
        assert hash(d0) != hash(d1)

        s1 = Scalar(name='s', dtype=np.int32)
        d2 = Dimension(name='d', spacing=s1)
        assert hash(d1) != hash(d2)

        d3 = Dimension(name='d', spacing=Constant(name='s1'))
        assert hash(d3) != hash(d0)
        assert hash(d3) != hash(d1)

    def test_sub_dimension(self):
        """Test that different SubDimensions have different hash value."""
        d0 = Dimension(name='d')
        d1 = Dimension(name='d', spacing=Scalar(name='s'))

        di0 = SubDimension.middle('di', d0, 1, 1)
        di1 = SubDimension.middle('di', d1, 1, 1)
        assert hash(di0) != hash(d0)
        assert hash(di0) != hash(di1)

        dl0 = SubDimension.left('dl', d0, 2)
        assert hash(dl0) != hash(di0)

    def test_conditional_dimension(self):
        """Test that different ConditionalDimensions have different hash value."""
        d0 = Dimension(name='d')
        s0 = Scalar(name='s')
        d1 = Dimension(name='d', spacing=s0)

        cd0 = ConditionalDimension(name='cd', parent=d0, factor=4)
        cd1 = ConditionalDimension(name='cd', parent=d0, factor=5)
        assert cd0 is not cd1
        assert hash(cd0) != hash(cd1)

        cd2 = ConditionalDimension(name='cd', parent=d0, factor=4, indirect=True)
        assert hash(cd0) != hash(cd2)

        cd3 = ConditionalDimension(name='cd', parent=d1, factor=4)
        assert hash(cd0) != hash(cd3)

        s1 = Scalar(name='s', dtype=np.int32)
        cd4 = ConditionalDimension(name='cd', parent=d0, factor=4, condition=s0 > 3)
        assert hash(cd0) != hash(cd4)

        cd5 = ConditionalDimension(name='cd', parent=d0, factor=4, condition=s1 > 3)
        assert hash(cd0) != hash(cd5)
        assert hash(cd4) != hash(cd5)

    def test_default_dimension(self):
        """Test that different DefaultDimensions have different hash value."""
        dd0 = DefaultDimension(name='dd')
        dd1 = DefaultDimension(name='dd')
        assert hash(dd0) != hash(dd1)

    @pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
    def test_function(self, FunctionType):
        """Test that different Functions have different hash value."""
        grid0 = Grid(shape=(3, 3))
        u0 = FunctionType(name='u', grid=grid0)
        grid1 = Grid(shape=(4, 4))
        u1 = FunctionType(name='u', grid=grid1)
        assert u0 is not u1
        assert hash(u0) != hash(u1)
        # Now with the same grid
        u2 = FunctionType(name='u', grid=grid0)
        assert u0 is not u2
        assert hash(u0) != hash(u2)

    @pytest.mark.parametrize('FunctionType', [SparseFunction, SparseTimeFunction])
    def test_sparse_function(self, FunctionType):
        """Test that different SparseFunctions have different hash value."""
        grid0 = Grid(shape=(3, 3))
        u0 = FunctionType(name='u', grid=grid0, npoint=1, nt=10)
        grid1 = Grid(shape=(4, 4))
        u1 = FunctionType(name='u', grid=grid1, npoint=1, nt=10)
        assert u0 is not u1
        assert hash(u0) != hash(u1)
        # Now with the same grid
        u2 = FunctionType(name='u', grid=grid0, npoint=1, nt=10)
        assert u0 is not u2
        assert hash(u0) != hash(u2)
        # Now with different number of sparse points
        u3 = FunctionType(name='u', grid=grid0, npoint=2, nt=10)
        assert u0 is not u3
        assert hash(u0) != hash(u3)
        # Now with different number of timesteps stored
        u4 = FunctionType(name='u', grid=grid0, npoint=1, nt=14)
        assert u0 is not u4
        assert hash(u0) != hash(u4)

    @pytest.mark.parametrize('FunctionType', [TensorFunction, TensorTimeFunction,
                                              VectorTimeFunction, VectorFunction])
    def test_tensor_hash(self, FunctionType):
        """Test that different Functions have different hash value."""
        grid0 = Grid(shape=(3, 3))
        u0 = FunctionType(name='u', grid=grid0)
        grid1 = Grid(shape=(4, 4))
        u1 = FunctionType(name='u', grid=grid1)
        assert u0 is not u1
        assert hash(u0) != hash(u1)
        # Now with the same grid
        u2 = FunctionType(name='u', grid=grid0)
        assert u0 is not u2
        assert hash(u0) != hash(u2)


class TestCaching(object):

    """
    Test the symbol cache infrastructure.
    """

    @pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
    def test_function(self, FunctionType):
        """Test that new u[x, y] instances don't cache"""
        grid = Grid(shape=(3, 4))
        u0 = FunctionType(name='u', grid=grid)
        u0.data[:] = 6.
        u1 = FunctionType(name='u', grid=grid)
        u1.data[:] = 2.
        assert np.allclose(u0.data, 6.)
        assert np.allclose(u1.data, 2.)

    @pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
    def test_function_same_indices(self, FunctionType):
        """Test caching of derived u[x, y] instance from derivative"""
        grid = Grid(shape=(3, 4))
        u0 = FunctionType(name='u', grid=grid)
        u0.data[:] = 6.
        # Pick u(x, y) and u(x + h_x, y) from derivative
        u1 = u0.dx.evaluate.args[1].args[2]
        u2 = u0.dx.evaluate.args[0].args[1]
        assert np.allclose(u1.data, 6.)
        assert np.allclose(u2.data, 6.)

    @pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
    def test_function_different_indices(self, FunctionType):
        """Test caching of u[x + h, y] instance from derivative"""
        grid = Grid(shape=(3, 4))
        u0 = FunctionType(name='u', grid=grid)
        u0.data[:] = 6.
        # Pick u[x + h, y] (different indices) from derivative
        u = u0.dx.evaluate.args[0].args[1]
        assert np.allclose(u.data, u0.data)

    @pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
    def test_function_duplicates(self, FunctionType):
        """Test caching of u[x + h, y] instance from derivative"""
        grid = Grid(shape=(3, 4))
        _cache_size = len(_SymbolCache)
        x = grid.dimensions[0]
        u0 = FunctionType(name='u', grid=grid)
        # u[x + h_x]
        uf = u0.subs({x: x + x.spacing})
        # u[x] shifting back from u[x + h_x]
        ub = uf.subs({x: x - x.spacing})
        # Make sure ub is u0
        assert ub is u0
        assert hash(ub) == hash(u0)
        # Three new cache entries: u, u(t,x,y), u(t, x+h_x, y)
        ncreated = 3
        assert len(_SymbolCache) == _cache_size + ncreated
        # shift again, no new entry should be created
        uf2 = ub.subs({x: x + x.spacing})
        assert uf is uf2
        assert len(_SymbolCache) == _cache_size + ncreated

    def test_symbols(self):
        """
        Test that ``Symbol(name='s') != Scalar(name='s') != Dimension(name='s')``.
        They all:

            * rely on the same caching mechanism
            * boil down to creating a sympy.Symbol
            * created with the same args/kwargs (``name='s'``)
        """
        sy = Symbol(name='s')
        sc = Scalar(name='s')
        d = Dimension(name='s')

        assert sy is not sc
        assert sc is not d
        assert sy is not d

        assert isinstance(sy, Symbol)
        assert isinstance(sc, Scalar)
        assert isinstance(d, Dimension)

    def test_symbols_args_vs_kwargs(self):
        """
        Unlike Functions, Symbols don't require the use of a kwarg to specify the name.
        This test basically checks that `Symbol('s') is Symbol(name='s')`, i.e. that we
        don't make any silly mistakes when it gets to compute the cache key.
        """
        v_arg = Symbol('v')
        v_kwarg = Symbol(name='v')
        assert v_arg is v_kwarg

        d_arg = Dimension('d100')
        d_kwarg = Dimension(name='d100')
        assert d_arg is d_kwarg

    def test_scalar(self):
        """
        Test that Scalars with same name but different attributes do not alias to
        the same Scalar. Conversely, if the name and the attributes are the same,
        they must alias to the same Scalar.
        """
        s0 = Scalar(name='s0')
        s1 = Scalar(name='s0')
        assert s0 is s1

        s2 = Scalar(name='s0', dtype=np.int32)
        assert s2 is not s1

        s3 = Scalar(name='s0', is_const=True)
        assert s3 is not s1

    def test_dimension(self):
        """
        Test that Dimensions with same name but different attributes do not alias to
        the same Dimension. Conversely, if the name and the attributes are the same,
        they must alias to the same Dimension.
        """
        d0 = Dimension(name='d')
        d1 = Dimension(name='d')
        assert d0 is d1

        s0 = Scalar(name='s0')
        s1 = Scalar(name='s1')

        d2 = Dimension(name='d', spacing=s0)
        d3 = Dimension(name='d', spacing=s1)
        assert d2 is not d3

        d4 = Dimension(name='d', spacing=s1)
        assert d3 is d4

        d5 = Dimension(name='d', spacing=Constant(name='s1'))
        assert d2 is not d5

    def test_conditional_dimension(self):
        """
        Test that ConditionalDimensions with same name but different attributes do not
        alias to the same ConditionalDimension. Conversely, if the name and the attributes
        are the same, they must alias to the same ConditionalDimension.
        """
        i = Dimension(name='i')
        ci0 = ConditionalDimension(name='ci', parent=i, factor=4)
        ci1 = ConditionalDimension(name='ci', parent=i, factor=4)
        assert ci0 is ci1

        ci2 = ConditionalDimension(name='ci', parent=i, factor=8)
        assert ci2 is not ci1

        ci3 = ConditionalDimension(name='ci', parent=i, factor=4, indirect=True)
        assert ci3 is not ci1

        s = Scalar(name='s')
        ci4 = ConditionalDimension(name='ci', parent=i, factor=4, condition=s > 3)
        assert ci4 is not ci1
        ci5 = ConditionalDimension(name='ci', parent=i, factor=4, condition=s > 3)
        assert ci5 is ci4

    def test_sub_dimension(self):
        """
        Test that SubDimensions with same name but different attributes do not
        alias to the same SubDimension. Conversely, if the name and the attributes
        are the same, they must alias to the same SubDimension.
        """
        x = Dimension('x')
        xi0 = SubDimension.middle('xi', x, 1, 1)
        xi1 = SubDimension.middle('xi', x, 1, 1)
        assert xi0 is xi1

        xl0 = SubDimension.left('xl', x, 2)
        xl1 = SubDimension.left('xl', x, 2)
        assert xl0 is xl1
        xl2asxi = SubDimension.left('xi', x, 2)
        assert xl2asxi is not xl1
        assert xl2asxi is not xi1

        xr0 = SubDimension.right('xr', x, 1)
        xr1 = SubDimension.right('xr', x, 1)
        assert xr0 is xr1

    def test_default_dimension(self):
        d = Dimension(name='d')
        dd0 = DefaultDimension(name='d')
        assert d is not dd0

        dd1 = DefaultDimension(name='d')
        assert dd0 is not dd1

    def test_constant_new(self):
        """Test that new u[x, y] instances don't cache"""
        u0 = Constant(name='u')
        u0.data = 6.
        u1 = Constant(name='u')
        u1.data = 2.
        assert u0.data == 6.
        assert u1.data == 2.

    def test_grid_objs(self):
        """
        Test that two different Grids use the same Symbols/Dimensions if possible
        (i.e., if already in cache). This is because objects such as spacing and origin
        are Scalars, which carry no value.
        """
        grid0 = Grid(shape=(4, 4))
        x0, y0 = grid0.dimensions
        ox0, oy0 = grid0.origin

        grid1 = Grid(shape=(8, 8))
        x1, y1 = grid1.dimensions
        ox1, oy1 = grid1.origin

        assert x0 is x1
        assert y0 is y1
        assert x0.spacing is x1.spacing
        assert y0.spacing is y1.spacing
        assert ox0 is ox1
        assert oy0 is oy1

    def test_special_symbols(self):
        """
        This test checks the singletonization, through the caching infrastructure,
        of the special symbols that an Operator may generate (e.g., `nthreads`).
        """
        grid = Grid(shape=(4, 4, 4))
        f = TimeFunction(name='f', grid=grid)
        sf = SparseTimeFunction(name='sf', grid=grid, npoint=1, nt=10)

        eqns = [Eq(f.forward, f + 1.)] + sf.inject(field=f.forward, expr=sf)

        opt = ('advanced', {'par-nested': 0, 'openmp': True})
        op0 = Operator(eqns, opt=opt)
        op1 = Operator(eqns, opt=opt)

        nthreads0, nthreads_nested0, nthreads_nonaffine0 =\
            [i for i in op0.input if isinstance(i, NThreadsBase)]
        nthreads1, nthreads_nested1, nthreads_nonaffine1 =\
            [i for i in op1.input if isinstance(i, NThreadsBase)]

        assert nthreads0 is nthreads1
        assert nthreads_nested0 is nthreads_nested1
        assert nthreads_nonaffine0 is nthreads_nonaffine1

        tid0 = ThreadID(op0.nthreads)
        tid1 = ThreadID(op0.nthreads)
        assert tid0 is tid1

        did0 = DeviceID()
        did1 = DeviceID()
        assert did0 is did1

        npt0 = NPThreads(name='npt', size=3)
        npt1 = NPThreads(name='npt', size=3)
        npt2 = NPThreads(name='npt', size=4)
        assert npt0 is npt1
        assert npt0 is not npt2

    def test_symbol_aliasing(self):
        """Test to assert that our aliasing cache isn't defeated by sympys
        non-aliasing symbol cache.

        For further explanation consider the symbol u[x, y] and it's first
        derivative in x, which includes the symbols u[x, y] and u[x + h, y].
        The two functions are aliased in devito's caching mechanism to allow
        multiple stencil indices pointing at the same data object u, but
        SymPy treats these two instances as separate functions and thus is
        allowed to delete one or the other when the cache is cleared.

        The test below asserts that u[x + h, y] is deleted, the data on u
        is still intact through our own caching mechanism."""

        # Ensure a clean cache to start with
        clear_cache()
        # FIXME: Currently not working, presumably due to our
        # failure to cache new instances?
        # assert(len(_SymbolCache) == 0)

        # Create first instance of u and fill its data
        grid = Grid(shape=(3, 4))
        u = Function(name='u', grid=grid)
        u.data[:] = 6.
        u_ref = weakref.ref(u.data)

        # Create u[x + h, y] and delete it again
        dx = u.dx  # Contains two u symbols: u[x, y] and u[x + h, y]
        del dx
        clear_cache()
        # FIXME: Unreliable cache sizes
        # assert len(_SymbolCache) == 1  # We still have a reference to u
        assert np.allclose(u.data, 6.)  # u.data is alive and well

        # Remove the final instance and ensure u.data got deallocated
        del u
        clear_cache()
        assert u_ref() is None

    def test_symbol_aliasing_reverse(self):
        """Test to assert that removing he original u[x, y] instance does
        not impede our alisaing cache or leaks memory.
        """

        # Ensure a clean cache to start with
        clear_cache()
        # FIXME: Currently not working, presumably due to our
        # failure to cache new instances?
        # assert(len(_SymbolCache) == 0)

        # Create first instance of u and fill its data
        grid = Grid(shape=(3, 4))
        u = Function(name='u', grid=grid)
        u.data[:] = 6.
        u_ref = weakref.ref(u.data)

        # Create derivative and delete orignal u[x, y]
        dx = u.dx
        del u
        clear_cache()
        # We still have a references to u
        # FIXME: Unreliable cache sizes
        # assert len(_SymbolCache) == 1
        # Ensure u[x + h, y] still holds valid data
        assert np.allclose(dx.evaluate.args[0].args[1].data, 6.)

        del dx
        clear_cache()
        # FIXME: Unreliable cache sizes
        # assert len(_SymbolCache) == 0  # We still have a reference to u_h
        assert u_ref() is None

    def test_clear_cache(self, operate_on_empty_cache, nx=1000, ny=1000):
        grid = Grid(shape=(nx, ny), dtype=np.float64)
        cache_size = len(_SymbolCache)

        for i in range(10):
            assert(len(_SymbolCache) == cache_size)

            Function(name='u', grid=grid, space_order=2)
            # Both u and u(inds) added to cache
            assert(len(_SymbolCache) == cache_size + 2)

            clear_cache()

    def test_clear_cache_with_Csymbol(self, operate_on_empty_cache, nx=1000, ny=1000):
        grid = Grid(shape=(nx, ny), dtype=np.float64)
        cache_size = len(_SymbolCache)

        u = Function(name='u', grid=grid, space_order=2)
        # Both u and u(inds) added to cache
        assert(len(_SymbolCache) == cache_size + 2)

        u._C_symbol
        # Cache size won't change since _C_symbol isn't cached by devito to
        # avoid circular references in the cache
        assert(len(_SymbolCache) == cache_size + 2)

    def test_clear_cache_with_alive_symbols(self, operate_on_empty_cache,
                                            nx=1000, ny=1000):
        """
        Test that `clear_cache` doesn't affect caching if an object is still alive.
        """
        grid = Grid(shape=(nx, ny), dtype=np.float64)

        f0 = Function(name='f', grid=grid, space_order=2)
        f1 = Function(name='f', grid=grid, space_order=2)

        # Obviously:
        assert f0 is not f1

        # And clearly, both still alive after a `clear_cache`
        clear_cache()
        assert f0 is not f1
        assert f0.grid.dimensions[0] is grid.dimensions[0]

        # Now we try with symbols
        s0 = Scalar(name='s')
        s1 = Scalar(name='s')

        # Clearly:
        assert s1 is s0

        clear_cache()
        s2 = Scalar(name='s')

        # s2 must still be s1/so, even after a clear_cache, as so/s1 are both alive!
        assert s2 is s1

        del s0
        del s1
        s3 = Scalar(name='s')

        # And obviously, still:
        assert s3 is s2

        cache_size = len(_SymbolCache)
        del s2
        del s3
        clear_cache()
        assert len(_SymbolCache) == cache_size - 1

    def test_sparse_function(self, operate_on_empty_cache):
        """Test caching of SparseFunctions and children objects."""
        grid = Grid(shape=(3, 3))

        init_cache_size = len(_SymbolCache)
        cur_cache_size = len(_SymbolCache)

        u = SparseFunction(name='u', grid=grid, npoint=1, nt=10)

        # created: u, u(inds), p_u, h_p_u, u_coords, u_coords(inds), d, h_d
        ncreated = 8
        assert len(_SymbolCache) == cur_cache_size + ncreated

        cur_cache_size = len(_SymbolCache)

        i = u.inject(expr=u, field=u)

        # created: ii_u_0*2 (Symbol and ConditionalDimension), ii_u_1*2, ii_u_2*2,
        # ii_u_3*2, px, py, posx, posy, u_coords (as indexified),
        ncreated = 2+2+2+2+2+1+1+1
        # Note that injection is now lazy so no new symbols should be created
        assert len(_SymbolCache) == cur_cache_size
        i.evaluate

        assert len(_SymbolCache) == cur_cache_size + ncreated

        # No new symbolic obejcts are created
        u.inject(expr=u, field=u)
        assert len(_SymbolCache) == cur_cache_size + ncreated

        # Let's look at clear_cache now
        del u
        del i
        clear_cache()
        # At this point, not all children objects have been cleared. In particular, the
        # ii_u_* Symbols are still alive, as well as p_u and h_p_u. This is because
        # in the first clear_cache they were still referenced by their "parent" objects
        # (e.g., ii_u_* by ConditionalDimensions, through `condition`)
        assert len(_SymbolCache) == init_cache_size + 8
        clear_cache()
        # Now we should be back to the original state
        assert len(_SymbolCache) == init_cache_size

    def test_after_indexification(self):
        """
        Test to assert that the SymPy cache retrieves the right Devito data object
        after indexification.
        """
        grid = Grid(shape=(4, 4, 4))
        u0 = Function(name='u', grid=grid, space_order=0)
        u1 = Function(name='u', grid=grid, space_order=1)
        u2 = Function(name='u', grid=grid, space_order=2)

        for i in [u0, u1, u2]:
            assert i.indexify().base.function.space_order ==\
                (i.indexify() + 1.).args[1].base.function.space_order

    def test_reinsertion_after_deletion(self, operate_on_empty_cache):
        """
        Test that dead weakrefs in the symbol cache do not cause any issues when
        objects with the same key/hash are reinserted.
        """
        d = Dimension(name='d')
        del d

        # `d` has just been deleted, but a weakref pointing to a dead object is still
        # in the symbol cache at this point; `h_d` is still in the cache too, dead too
        assert len(_SymbolCache) == 2
        assert all(i() is None for i in _SymbolCache.values())

        d = Dimension(name='d')  # noqa
        assert len(_SymbolCache) == 2
        assert all(i() is not None for i in _SymbolCache.values())

    @pytest.mark.parametrize('FunctionType', [VectorFunction, TensorFunction,
                                              VectorTimeFunction, TensorTimeFunction])
    def test_tensor_different_indices(self, FunctionType):
        """Test caching of u[x + h, y] instance from derivative"""
        grid = Grid(shape=(3, 4))
        u0 = FunctionType(name='u', grid=grid)
        for s in u0:
            s.data[:] = 6.
        # Pick u[x + h, y] (different indices) from derivative
        u = u0.dx.evaluate[0].args[0].args[1]
        assert np.allclose(u.data, u0[0].data)

    @pytest.mark.parametrize('FunctionType', [VectorFunction, TensorFunction,
                                              VectorTimeFunction, TensorTimeFunction])
    def test_tensor_same_indices(self, FunctionType):
        """Test caching of derived u[x, y] instance from derivative"""
        grid = Grid(shape=(3, 4))
        u0 = FunctionType(name='u', grid=grid)
        for s in u0:
            s.data[:] = 6.
        # Pick u(x, y) and u(x + h_x, y) from derivative
        u1 = u0.dx.evaluate[0].args[1].args[2]
        u2 = u0.dx.evaluate[1].args[0].args[1]
        assert np.allclose(u1.data, 6.)
        assert np.allclose(u2.data, 6.)

    @pytest.mark.parametrize('FunctionType', [VectorFunction, TensorFunction,
                                              VectorTimeFunction, TensorTimeFunction])
    def test_tensor_new(self, FunctionType):
        """Test that new u[x, y] instances don't cache"""
        grid = Grid(shape=(3, 4))
        u0 = FunctionType(name='u', grid=grid)
        for s in u0:
            s.data[:] = 6.
        u1 = FunctionType(name='u', grid=grid)
        for s in u1:
            s.data[:] = 2.
        assert np.all(np.allclose(s.data, 6.) for s in u0)


class TestMemoryLeaks(object):

    """
    Tests ensuring there are no memory leaks.
    """

    def test_operator_leakage_function(self):
        """
        Test to ensure that Operator creation does not cause memory leaks for
        (Time)Functions.
        """
        grid = Grid(shape=(5, 6))
        f = Function(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)

        # Take weakrefs to test whether symbols are dead or alive
        w_f = weakref.ref(f)
        w_g = weakref.ref(g)

        # Create operator and delete everything again
        op = Operator(Eq(f, 2 * g))
        w_op = weakref.ref(op)
        del op
        del f
        del g
        clear_cache()

        # Test whether things are still hanging around
        assert w_f() is None
        assert w_g() is None
        assert w_op() is None

    def test_operator_leakage_sparse(self):
        """
        Test to ensure that Operator creation does not cause memory leaks for
        SparseTimeFunctions.
        """
        grid = Grid(shape=(5, 6))
        a = Function(name='a', grid=grid)
        s = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=1)
        w_a = weakref.ref(a)
        w_s = weakref.ref(s)

        # Create operator and delete everything again
        op = Operator(s.interpolate(a))
        w_op = weakref.ref(op)
        del op
        del s
        del a
        clear_cache()

        # Test whether things are still hanging around
        assert w_a() is None
        assert w_s() is None
        assert w_op() is None

    def test_solve(self, operate_on_empty_cache):
        """
        Test to ensure clear_cache wipes out *all of* sympy caches. ``sympy.solve``,
        in particular, relies on a series of private caches that must be purged too
        (calling sympy's clear_cache() API function isn't enough).
        """
        grid = Grid(shape=(4,))

        u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)

        eqn = Eq(u.dt, u.dx2)
        solve(eqn, u.forward)

        del u
        del eqn
        del grid

        # We only deleted `u`, however we also cache shifted version created by the
        # finite difference (u.dt, u.dx2). In this case we have three extra references
        # to u(t + dt), u(x - h_x) and u(x + h_x) that have to be cleared.
        # Then `u` points to the various Dimensions, the Dimensions point to the various
        # spacing symbols, hence, we need four sweeps to clear up the cache.
        assert len(_SymbolCache) == 14
        clear_cache()
        assert len(_SymbolCache) == 9
        clear_cache()
        assert len(_SymbolCache) == 3
        clear_cache()
        assert len(_SymbolCache) == 1
        clear_cache()
        assert len(_SymbolCache) == 0
