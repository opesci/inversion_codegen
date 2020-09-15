from collections import namedtuple
from ctypes import POINTER, Structure, c_void_p, c_int, cast, byref
from functools import wraps, reduce
from math import ceil
from operator import mul

import numpy as np
import sympy
from psutil import virtual_memory
from cached_property import cached_property
from cgen import Struct, Value

from devito.builtins import assign
from devito.data import (DOMAIN, OWNED, HALO, NOPAD, FULL, LEFT, CENTER, RIGHT,
                         Data, default_allocator)
from devito.exceptions import InvalidArgument
from devito.logger import debug, warning
from devito.mpi import MPI
from devito.parameters import configuration
from devito.symbolics import FieldFromPointer
from devito.finite_differences import Differentiable, generate_fd_shortcuts
from devito.tools import (ReducerMap, as_tuple, flatten, is_integer,
                          ctypes_to_cstr, memoized_meth, dtype_to_ctype)
from devito.types.dimension import Dimension
from devito.types.args import ArgProvider
from devito.types.caching import CacheManager
from devito.types.basic import AbstractFunction, Size
from devito.types.utils import Buffer, DimensionTuple, NODE, CELL

__all__ = ['Function', 'TimeFunction']


RegionMeta = namedtuple('RegionMeta', 'offset size')


class DiscreteFunction(AbstractFunction, ArgProvider, Differentiable):

    """
    Tensor symbol representing a discrete function in symbolic equations.
    Unlike an Array, a DiscreteFunction carries data.

    Notes
    -----
    Users should not instantiate this class directly. Use Function or
    SparseFunction (or their subclasses) instead.
    """

    # Required by SymPy, otherwise the presence of __getitem__ will make SymPy
    # think that a DiscreteFunction is actually iterable, thus breaking many of
    # its key routines (e.g., solve)
    _iterable = False

    is_Function = False
    is_Input = True
    is_DiscreteFunction = True
    is_Tensor = True

    def __init_finalize__(self, *args, **kwargs):
        # A `Distributor` to handle domain decomposition (only relevant for MPI)
        self._distributor = self.__distributor_setup__(**kwargs)

        # Staggering metadata
        self._staggered = self.__staggered_setup__(**kwargs)

        # Now that *all* __X_setup__ hooks have been called, we can let the
        # superclass constructor do its job
        super(DiscreteFunction, self).__init_finalize__(*args, **kwargs)

        # There may or may not be a `Grid` attached to the DiscreteFunction
        self._grid = kwargs.get('grid')

        # Symbolic (finite difference) coefficients
        self._coefficients = kwargs.get('coefficients', 'standard')
        if self._coefficients not in ('standard', 'symbolic'):
            raise ValueError("coefficients must be `standard` or `symbolic`")

        # Data-related properties and data initialization
        self._data = None
        self._first_touch = kwargs.get('first_touch', configuration['first-touch'])
        self._allocator = kwargs.get('allocator', default_allocator())
        initializer = kwargs.get('initializer')
        if initializer is None or callable(initializer):
            # Initialization postponed until the first access to .data
            self._initializer = initializer
        elif isinstance(initializer, (np.ndarray, list, tuple)):
            # Allocate memory and initialize it. Note that we do *not* hold
            # a reference to the user-provided buffer
            self._initializer = None
            if len(initializer) > 0:
                self.data_with_halo[:] = initializer
            else:
                # This is a corner case -- we might get here, for example, when
                # running with MPI and some processes get 0-size arrays after
                # domain decomposition. We touch the data anyway to avoid the
                # case ``self._data is None``
                self.data
        else:
            raise ValueError("`initializer` must be callable or buffer, not %s"
                             % type(initializer))

    def __eq__(self, other):
        # The only possibility for two DiscreteFunctions to be considered equal
        # is that they are indeed the same exact object
        return self is other

    _subs = Differentiable._subs

    __hash__ = AbstractFunction.__hash__  # Required since we're overriding __eq__

    def _allocate_memory(func):
        """Allocate memory as a Data."""
        @wraps(func)
        def wrapper(self):
            if self._data is None:
                debug("Allocating memory for %s%s" % (self.name, self.shape_allocated))

                # Clear up both SymPy and Devito caches to drop unreachable data
                CacheManager.clear(force=False)

                # Allocate the actual data object
                self._data = Data(self.shape_allocated, self.dtype,
                                  modulo=self._mask_modulo, allocator=self._allocator,
                                  distributor=self._distributor)

                # Initialize data
                if self._first_touch:
                    assign(self, 0)
                if callable(self._initializer):
                    if self._first_touch:
                        warning("`first touch` together with `initializer` causing "
                                "redundant data initialization")
                    try:
                        self._initializer(self.data_with_halo)
                    except ValueError:
                        # Perhaps user only wants to initialise the physical domain
                        self._initializer(self.data)
                else:
                    self.data_with_halo.fill(0)

            return func(self)
        return wrapper

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        dtype = kwargs.get('dtype')
        if dtype is not None:
            return dtype
        elif grid is not None:
            return grid.dtype
        else:
            return np.float32

    def __staggered_setup__(self, **kwargs):
        """
        Setup staggering-related metadata. This method assigns:

            * 0 to non-staggered dimensions;
            * 1 to staggered dimensions.
        """
        staggered = kwargs.get('staggered', None)
        if staggered is CELL:
            staggered = self.dimensions
        return staggered

    def __distributor_setup__(self, **kwargs):
        grid = kwargs.get('grid')
        # There may or may not be a `Distributor`. In the latter case, the
        # DiscreteFunction is to be considered "local" to each MPI rank
        return kwargs.get('distributor') if grid is None else grid.distributor

    @cached_property
    def _functions(self):
        return {self.function}

    @property
    def _data_buffer(self):
        """
        Reference to the data. Unlike :attr:`data` and :attr:`data_with_halo`,
        this *never* returns a view of the data. This method is for internal use only.
        """
        return self._data_allocated

    @property
    def _data_alignment(self):
        return self._allocator.guaranteed_alignment

    @property
    def _mem_external(self):
        return True

    @property
    def grid(self):
        """The Grid on which the discretization occurred."""
        return self._grid

    @property
    def staggered(self):
        return self._staggered

    @property
    def coefficients(self):
        """Form of the coefficients of the function."""
        return self._coefficients

    @cached_property
    def _coeff_symbol(self):
        if self.coefficients == 'symbolic':
            return sympy.Function('W')
        else:
            raise ValueError("Function was not declared with symbolic "
                             "coefficients.")

    @cached_property
    def shape(self):
        """
        Shape of the domain region. The domain constitutes the area of the
        data written to by an Operator.

        Notes
        -----
        In an MPI context, this is the *local* domain region shape.
        """
        return self._shape

    @cached_property
    def shape_domain(self):
        """
        Shape of the domain region. The domain constitutes the area of the
        data written to by an Operator.

        Notes
        -----
        In an MPI context, this is the *local* domain region shape.
        Alias to ``self.shape``.
        """
        return self.shape

    @cached_property
    def shape_with_halo(self):
        """
        Shape of the domain+outhalo region. The outhalo is the region
        surrounding the domain that may be read by an Operator.

        Notes
        -----
        In an MPI context, this is the *local* with_halo region shape.
        Further, note that the outhalo of inner ranks is typically empty, while
        the outhalo of boundary ranks contains a number of elements depending
        on the rank position in the decomposed grid (corner, side, ...).
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape, self._size_outhalo))

    _shape_with_outhalo = shape_with_halo

    @cached_property
    def _shape_with_inhalo(self):
        """
        Shape of the domain+inhalo region. The inhalo region comprises the
        outhalo as well as any additional "ghost" layers for MPI halo
        exchanges. Data in the inhalo region are exchanged when running
        Operators to maintain consistent values as in sequential runs.

        Notes
        -----
        Typically, this property won't be used in user code, but it may come
        in handy for testing or debugging
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape, self._halo))

    @cached_property
    def shape_allocated(self):
        """
        Shape of the allocated data. It includes the domain and inhalo regions,
        as well as any additional padding surrounding the halo.

        Notes
        -----
        In an MPI context, this is the *local* with_halo region shape.
        """
        return tuple(j + i + k for i, (j, k) in zip(self._shape_with_inhalo,
                                                    self._padding))

    @cached_property
    def shape_global(self):
        """
        Global shape of the domain region. The domain constitutes the area of
        the data written to by an Operator.

        Notes
        -----
        In an MPI context, this is the *global* domain region shape, which is
        therefore identical on all MPI ranks.
        """
        if self.grid is None:
            return self.shape
        retval = []
        for d, s in zip(self.dimensions, self.shape):
            size = self.grid.dimension_map.get(d)
            retval.append(size.glb if size is not None else s)
        return tuple(retval)

    _offset_inhalo = AbstractFunction._offset_halo
    _size_inhalo = AbstractFunction._size_halo

    @cached_property
    def _size_outhalo(self):
        """Number of points in the outer halo region."""

        if self._distributor is None:
            # Computational domain is not distributed and hence the outhalo
            # and inhalo correspond
            return self._size_inhalo

        left = [abs(min(i.loc_abs_min-i.glb_min-j, 0)) if i and not i.loc_empty else 0
                for i, j in zip(self._decomposition, self._size_inhalo.left)]
        right = [max(i.loc_abs_max+j-i.glb_max, 0) if i and not i.loc_empty else 0
                 for i, j in zip(self._decomposition, self._size_inhalo.right)]

        sizes = tuple(Size(i, j) for i, j in zip(left, right))

        if self._distributor.is_parallel and (any(left) > 0 or any(right)) > 0:
            try:
                warning_msg = """A space order of {0} and a halo size of {1} has been
                                 set but the current rank ({2}) has a domain size of
                                 only {3}""".format(self._space_order,
                                                    max(self._size_inhalo),
                                                    self._distributor.myrank,
                                                    min(self.grid.shape_local))
                if not self._distributor.is_boundary_rank:
                    warning(warning_msg)
                else:
                    for i, j, k, l in zip(left, right, self._distributor.mycoords,
                                          self._distributor.topology):
                        if l > 1 and ((j > 0 and k == 0) or (i > 0 and k == l-1)):
                            warning(warning_msg)
                            break
            except AttributeError:
                pass

        return DimensionTuple(*sizes, getters=self.dimensions, left=left, right=right)

    @property
    def size_allocated(self):
        """
        The number of elements this object is expected to store in memory.
        Note that this would need to be combined with self.dtype to give the actual
        size in bytes.
        """
        return reduce(mul, self.shape_allocated)

    @cached_property
    def _mask_modulo(self):
        """Boolean mask telling which Dimensions support modulo-indexing."""
        return tuple(True if i.is_Stepping else False for i in self.dimensions)

    @cached_property
    def _mask_domain(self):
        """Slice-based mask to access the domain region of the allocated data."""
        return tuple(slice(i, j) for i, j in
                     zip(self._offset_domain, self._offset_halo.right))

    @cached_property
    def _mask_inhalo(self):
        """Slice-based mask to access the domain+inhalo region of the allocated data."""
        return tuple(slice(i.left, i.right + j.right) for i, j in
                     zip(self._offset_inhalo, self._size_inhalo))

    @cached_property
    def _mask_outhalo(self):
        """Slice-based mask to access the domain+outhalo region of the allocated data."""
        return tuple(slice(i.start - j.left, i.stop and i.stop + j.right or None)
                     for i, j in zip(self._mask_domain, self._size_outhalo))

    @cached_property
    def _decomposition(self):
        """
        Tuple of Decomposition objects, representing the domain decomposition.
        None is used as a placeholder for non-decomposed Dimensions.
        """
        if self._distributor is None:
            return (None,)*self.ndim
        mapper = {d: self._distributor.decomposition[d] for d in self._dist_dimensions}
        return tuple(mapper.get(d) for d in self.dimensions)

    @cached_property
    def _decomposition_outhalo(self):
        """
        Tuple of Decomposition objects, representing the domain+outhalo
        decomposition. None is used as a placeholder for non-decomposed Dimensions.
        """
        if self._distributor is None:
            return (None,)*self.ndim
        return tuple(v.reshape(*self._size_inhalo[d]) if v is not None else v
                     for d, v in zip(self.dimensions, self._decomposition))

    @property
    def data(self):
        """
        The domain data values, as a numpy.ndarray.

        Elements are stored in row-major format.

        Notes
        -----
        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use :meth:`data_ro`
        instead.
        """
        return self.data_domain

    @property
    @_allocate_memory
    def data_domain(self):
        """
        The domain data values.

        Elements are stored in row-major format.

        Notes
        -----
        Alias to ``self.data``.

        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use
        :meth:`data_ro_domain` instead.
        """
        self._is_halo_dirty = True
        return self._data._global(self._mask_domain, self._decomposition)

    @property
    @_allocate_memory
    def data_with_halo(self):
        """
        The domain+outhalo data values.

        Elements are stored in row-major format.

        Notes
        -----
        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use
        :meth:`data_ro_with_halo` instead.
        """
        self._is_halo_dirty = True
        self._halo_exchange()
        return self._data._global(self._mask_outhalo, self._decomposition_outhalo)

    _data_with_outhalo = data_with_halo

    @property
    @_allocate_memory
    def _data_with_inhalo(self):
        """
        The domain+inhalo data values.

        Elements are stored in row-major format.

        Notes
        -----
        This accessor does *not* support global indexing.

        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use
        :meth:`data_ro_with_inhalo` instead.

        Typically, this accessor won't be used in user code to set or read data
        values. Instead, it may come in handy for testing or debugging
        """
        self._is_halo_dirty = True
        self._halo_exchange()
        return np.asarray(self._data[self._mask_inhalo])

    @property
    @_allocate_memory
    def _data_allocated(self):
        """
        The allocated data values, that is domain+inhalo+padding.

        Elements are stored in row-major format.

        Notes
        -----
        This accessor does *not* support global indexing.

        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use
        :meth:`data_ro_allocated` instead.

        Typically, this accessor won't be used in user code to set or read data
        values. Instead, it may come in handy for testing or debugging
        """
        self._is_halo_dirty = True
        self._halo_exchange()
        return np.asarray(self._data)

    def _data_in_region(self, region, dim, side):
        """
        The data values in a given region.

        Parameters
        ----------
        region : DataRegion
            The data region of interest (e.g., OWNED, HALO) for which a view
            is produced.
        dim : Dimension
            The dimension of interest.
        side : DataSide
            The side of interest (LEFT, RIGHT).

        Notes
        -----
        This accessor does *not* support global indexing.

        With this accessor you are claiming that you will modify the values you
        get back.

        Typically, this accessor won't be used in user code to set or read
        data values.
        """
        self._is_halo_dirty = True
        offset = getattr(getattr(self, '_offset_%s' % region.name)[dim], side.name)
        size = getattr(getattr(self, '_size_%s' % region.name)[dim], side.name)
        index_array = [slice(offset, offset+size) if d is dim else slice(None)
                       for d in self.dimensions]
        return np.asarray(self._data[index_array])

    @property
    @_allocate_memory
    def data_ro_domain(self):
        """Read-only view of the domain data values."""
        view = self._data._global(self._mask_domain, self._decomposition)
        view.setflags(write=False)
        return view

    @property
    @_allocate_memory
    def data_ro_with_halo(self):
        """Read-only view of the domain+outhalo data values."""
        view = self._data._global(self._mask_outhalo, self._decomposition_outhalo)
        view.setflags(write=False)
        return view

    _data_ro_with_outhalo = data_ro_with_halo

    @property
    @_allocate_memory
    def _data_ro_with_inhalo(self):
        """
        Read-only view of the domain+inhalo data values.

        Notes
        -----
        This accessor does *not* support global indexing.
        """
        view = self._data[self._mask_inhalo]
        view.setflags(write=False)
        return np.asarray(view)

    @property
    @_allocate_memory
    def _data_ro_allocated(self):
        """
        Read-only view of the domain+inhalo+padding data values.

        Notes
        -----
        This accessor does *not* support global indexing.
        """
        view = self._data
        view.setflags(write=False)
        return np.asarray(view)

    @cached_property
    def local_indices(self):
        """
        Tuple of slices representing the global indices that logically
        belong to the calling MPI rank.

        Notes
        -----
        Given a Function ``f(x, y)`` with shape ``(nx, ny)``, when *not* using
        MPI this property will return ``(slice(0, nx-1), slice(0, ny-1))``. On
        the other hand, when MPI is used, the local ranges depend on the domain
        decomposition, which is carried by ``self.grid``.
        """
        if self._distributor is None:
            return tuple(slice(0, s) for s in self.shape)
        else:
            return tuple(self._distributor.glb_slices.get(d, slice(0, s))
                         for s, d in zip(self.shape, self.dimensions))

    @cached_property
    def space_dimensions(self):
        """Tuple of Dimensions defining the physical space."""
        return tuple(d for d in self.dimensions if d.is_Space)

    @cached_property
    def _dist_dimensions(self):
        """Tuple of MPI-distributed Dimensions."""
        if self._distributor is None:
            return ()
        return tuple(d for d in self.dimensions if d in self._distributor.dimensions)

    @property
    def initializer(self):
        if self._data is not None:
            return self.data_with_halo.view(np.ndarray)
        else:
            return self._initializer

    _C_structname = 'dataobj'
    _C_typename = 'struct %s *' % _C_structname
    _C_field_data = 'data'
    _C_field_size = 'size'
    _C_field_nopad_size = 'npsize'
    _C_field_domain_size = 'dsize'
    _C_field_halo_size = 'hsize'
    _C_field_halo_ofs = 'hofs'
    _C_field_owned_ofs = 'oofs'

    _C_typedecl = Struct(_C_structname,
                         [Value('%srestrict' % ctypes_to_cstr(c_void_p), _C_field_data),
                          Value(ctypes_to_cstr(POINTER(c_int)), _C_field_size),
                          Value(ctypes_to_cstr(POINTER(c_int)), _C_field_nopad_size),
                          Value(ctypes_to_cstr(POINTER(c_int)), _C_field_domain_size),
                          Value(ctypes_to_cstr(POINTER(c_int)), _C_field_halo_size),
                          Value(ctypes_to_cstr(POINTER(c_int)), _C_field_halo_ofs),
                          Value(ctypes_to_cstr(POINTER(c_int)), _C_field_owned_ofs)])

    _C_ctype = POINTER(type(_C_structname, (Structure,),
                            {'_fields_': [(_C_field_data, c_void_p),
                                          (_C_field_size, POINTER(c_int)),
                                          (_C_field_nopad_size, POINTER(c_int)),
                                          (_C_field_domain_size, POINTER(c_int)),
                                          (_C_field_halo_size, POINTER(c_int)),
                                          (_C_field_halo_ofs, POINTER(c_int)),
                                          (_C_field_owned_ofs, POINTER(c_int))]}))

    def _C_make_dataobj(self, data):
        """
        A ctypes object representing the DiscreteFunction that can be passed to
        an Operator.
        """
        dataobj = byref(self._C_ctype._type_())
        dataobj._obj.data = data.ctypes.data_as(c_void_p)
        dataobj._obj.size = (c_int*self.ndim)(*data.shape)
        # MPI-related fields
        dataobj._obj.npsize = (c_int*self.ndim)(*[i - sum(j) for i, j in
                                                  zip(data.shape, self._size_padding)])
        dataobj._obj.dsize = (c_int*self.ndim)(*self._size_domain)
        dataobj._obj.hsize = (c_int*(self.ndim*2))(*flatten(self._size_halo))
        dataobj._obj.hofs = (c_int*(self.ndim*2))(*flatten(self._offset_halo))
        dataobj._obj.oofs = (c_int*(self.ndim*2))(*flatten(self._offset_owned))

        # stash a reference to the array on _obj, so we don't let it get freed
        # while we hold onto _obj
        dataobj._obj.underlying_array = data

        return dataobj

    def _C_as_ndarray(self, dataobj):
        """Cast the data carried by a DiscreteFunction dataobj to an ndarray."""
        shape = tuple(dataobj._obj.size[i] for i in range(self.ndim))
        ctype_1d = dtype_to_ctype(self.dtype) * int(reduce(mul, shape))
        buf = cast(dataobj._obj.data, POINTER(ctype_1d)).contents
        return np.frombuffer(buf, dtype=self.dtype).reshape(shape)

    @memoized_meth
    def _C_make_index(self, dim, side=None):
        # Depends on how fields are populated in `_C_make_dataobj`
        idx = self.dimensions.index(dim)
        if side is not None:
            idx = idx*2 + (0 if side is LEFT else 1)
        return idx

    @memoized_meth
    def _C_get_field(self, region, dim, side=None):
        """Symbolic representation of a given data region."""
        ffp = lambda f, i: FieldFromPointer("%s[%d]" % (f, i), self._C_name)
        if region is DOMAIN:
            offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, LEFT))
            size = ffp(self._C_field_domain_size, self._C_make_index(dim))
        elif region is OWNED:
            if side is LEFT:
                offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, LEFT))
                size = ffp(self._C_field_halo_size, self._C_make_index(dim, RIGHT))
            elif side is CENTER:
                # Note: identical to region=HALO, side=CENTER
                offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, LEFT))
                size = ffp(self._C_field_domain_size, self._C_make_index(dim))
            else:
                offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, RIGHT))
                size = ffp(self._C_field_halo_size, self._C_make_index(dim, LEFT))
        elif region is HALO:
            if side is LEFT:
                offset = ffp(self._C_field_halo_ofs, self._C_make_index(dim, LEFT))
                size = ffp(self._C_field_halo_size, self._C_make_index(dim, LEFT))
            elif side is CENTER:
                # Note: identical to region=OWNED, side=CENTER
                offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, LEFT))
                size = ffp(self._C_field_domain_size, self._C_make_index(dim))
            else:
                offset = ffp(self._C_field_halo_ofs, self._C_make_index(dim, RIGHT))
                size = ffp(self._C_field_halo_size, self._C_make_index(dim, RIGHT))
        elif region is NOPAD:
            offset = ffp(self._C_field_halo_ofs, self._C_make_index(dim, LEFT))
            size = ffp(self._C_field_nopad_size, self._C_make_index(dim))
        elif region is FULL:
            offset = 0
            size = ffp(self._C_field_size, self._C_make_index(dim))
        else:
            raise ValueError("Unknown region `%s`" % str(region))

        return RegionMeta(offset, size)

    def _halo_exchange(self):
        """Perform the halo exchange with the neighboring processes."""
        if not MPI.Is_initialized() or MPI.COMM_WORLD.size == 1:
            # Nothing to do
            return
        if MPI.COMM_WORLD.size > 1 and self._distributor is None:
            raise RuntimeError("`%s` cannot perform a halo exchange as it has "
                               "no Grid attached" % self.name)

        neighborhood = self._distributor.neighborhood
        comm = self._distributor.comm

        for d in self._dist_dimensions:
            for i in [LEFT, RIGHT]:
                # Get involved peers
                dest = neighborhood[d][i]
                source = neighborhood[d][i.flip()]

                # Gather send data
                data = self._data_in_region(OWNED, d, i)
                sendbuf = np.ascontiguousarray(data)

                # Setup recv buffer
                shape = self._data_in_region(HALO, d, i.flip()).shape
                recvbuf = np.ndarray(shape=shape, dtype=self.dtype)

                # Communication
                comm.Sendrecv(sendbuf, dest=dest, recvbuf=recvbuf, source=source)

                # Scatter received data
                if recvbuf is not None and source != MPI.PROC_NULL:
                    self._data_in_region(HALO, d, i.flip())[:] = recvbuf

        self._is_halo_dirty = False

    @property
    def _arg_names(self):
        """Tuple of argument names introduced by this function."""
        return (self.name,)

    # NOTE: This _arg_defaults, unlike many others, cannot be memoized
    # since it leads to halo's not being updated when required. See
    # test_mpi: test_haloupdate_multi_op
    def _arg_defaults(self, alias=None):
        """
        A map of default argument values defined by this symbol.

        Parameters
        ----------
        alias : DiscreteFunction, optional
            To bind the argument values to different names.
        """
        key = alias or self
        args = ReducerMap({key.name: self._data_buffer})

        # Collect default dimension arguments from all indices
        for i, s in zip(key.dimensions, self.shape):
            args.update(i._arg_defaults(_min=0, size=s))

        # Add MPI-related data structures
        if self.grid is not None:
            args.update(self.grid._arg_defaults())

        return args

    def _arg_values(self, **kwargs):
        """
        A map of argument values after evaluating user input. If no
        user input is provided, return a default value.

        Parameters
        ----------
        **kwargs
            Dictionary of user-provided argument overrides.
        """
        # Add value override for own data if it is provided, otherwise
        # use defaults
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, DiscreteFunction):
                # Set new values and re-derive defaults
                values = new._arg_defaults(alias=self).reduce_all()
            else:
                # We've been provided a pure-data replacement (array)
                values = {self.name: new}
                # Add value overrides for all associated dimensions
                for i, s in zip(self.dimensions, new.shape):
                    size = s - sum(self._size_nodomain[i])
                    values.update(i._arg_defaults(size=size))
                # Add MPI-related data structures
                if self.grid is not None:
                    values.update(self.grid._arg_defaults())
        else:
            values = self._arg_defaults(alias=self).reduce_all()

        return values

    def _arg_check(self, args, intervals):
        """
        Check that ``args`` contains legal runtime values bound to ``self``.

        Raises
        ------
        InvalidArgument
            If, given the runtime values ``args``, an out-of-bounds array
            access would be performed, or if shape/dtype don't match with
            self's shape/dtype.
        """
        if self.name not in args:
            raise InvalidArgument("No runtime value for `%s`" % self.name)
        key = args[self.name]
        if len(key.shape) != self.ndim:
            raise InvalidArgument("Shape %s of runtime value `%s` does not match "
                                  "dimensions %s" %
                                  (key.shape, self.name, self.dimensions))
        if key.dtype != self.dtype:
            warning("Data type %s of runtime value `%s` does not match the "
                    "Function data type %s" % (key.dtype, self.name, self.dtype))
        for i, s in zip(self.dimensions, key.shape):
            i._arg_check(args, s, intervals[i])

    def _arg_as_ctype(self, args, alias=None):
        key = alias or self
        return ReducerMap({key.name: self._C_make_dataobj(args[key.name])})

    # Pickling support
    _pickle_kwargs = AbstractFunction._pickle_kwargs +\
        ['grid', 'staggered', 'initializer']


class Function(DiscreteFunction):

    """
    Tensor symbol representing a discrete function in symbolic equations.

    A Function carries multi-dimensional data and provides operations to create
    finite-differences approximations.

    A Function encapsulates space-varying data; for data that also varies in time,
    use TimeFunction instead.

    Parameters
    ----------
    name : str
        Name of the symbol.
    grid : Grid, optional
        Carries shape, dimensions, and dtype of the Function. When grid is not
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
        Define how the Function is staggered.
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

    Examples
    --------
    Creation

    >>> from devito import Grid, Function
    >>> grid = Grid(shape=(4, 4))
    >>> f = Function(name='f', grid=grid)
    >>> f
    f(x, y)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> g
    g(x, y)

    First-order derivatives through centered finite-difference approximations

    >>> f.dx
    Derivative(f(x, y), x)
    >>> f.dy
    Derivative(f(x, y), y)
    >>> g.dx
    Derivative(g(x, y), x)
    >>> (f + g).dx
    Derivative(f(x, y) + g(x, y), x)

    First-order derivatives through left/right finite-difference approximations

    >>> f.dxl
    Derivative(f(x, y), x)

    Note that the fact that it's a left-derivative isn't captured in the representation.
    However, upon derivative expansion, this becomes clear

    >>> f.dxl.evaluate
    f(x, y)/h_x - f(x - h_x, y)/h_x
    >>> f.dxr
    Derivative(f(x, y), x)

    Second-order derivative through centered finite-difference approximation

    >>> g.dx2
    Derivative(g(x, y), (x, 2))

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy
    uses ``*args`` to (re-)create the dimension arguments of the symbolic object.
    """

    is_Function = True

    def _cache_meta(self):
        # Attach additional metadata to self's cache entry
        return {'nbytes': self.size}

    def __init_finalize__(self, *args, **kwargs):
        super(Function, self).__init_finalize__(*args, **kwargs)

        # Space order
        space_order = kwargs.get('space_order', 1)
        if isinstance(space_order, int):
            self._space_order = space_order
        elif isinstance(space_order, tuple) and len(space_order) == 3:
            self._space_order, _, _ = space_order
        else:
            raise TypeError("`space_order` must be int or 3-tuple of ints")

        # Dynamically add derivative short-cuts
        self._fd = generate_fd_shortcuts(self)

        # Check if defined on a subdomain
        self._subdomain = kwargs.get('subdomain', None)

        # Flag whether it is a parameter or a variable.
        # Used at operator evaluation to evaluate the Function at the
        # variable location (i.e. if the variable is staggered in x the
        # parameter has to be computed at x + hx/2)
        self._is_parameter = kwargs.get('parameter', False)

    @cached_property
    def _fd_priority(self):
        return 1 if self.staggered in [NODE, None] else 2

    @property
    def is_parameter(self):
        return self._is_parameter

    def _eval_at(self, func):
        if not self.is_parameter or self.staggered == func.staggered:
            return self
        mapper = {self.indices_ref[d]: func.indices_ref[d]
                  for d in self.dimensions
                  if self.indices_ref[d] is not func.indices_ref[d]}
        if mapper:
            return self.subs(mapper)
        return self

    @classmethod
    def __indices_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        subdomain = kwargs.get('subdomain')
        dimensions = kwargs.get('dimensions')
        if grid is None:
            if dimensions is None:
                raise TypeError("Need either `grid` or `dimensions`")
        elif dimensions is None:
            if subdomain is None:
                dimensions = grid.dimensions
            else:
                dimensions = subdomain.dimensions

        # Staggered indices
        staggered = kwargs.get("staggered", None)
        if staggered in [CELL, NODE]:
            staggered_indices = dimensions
        else:
            mapper = {d: d for d in dimensions}
            for s in as_tuple(staggered):
                c, s = s.as_coeff_Mul()
                mapper.update({s: s + c * s.spacing/2})
            staggered_indices = mapper.values()

        return tuple(dimensions), tuple(staggered_indices)

    @property
    def is_Staggered(self):
        return self.staggered is not None

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        subdomain = kwargs.get('subdomain')
        dimensions = kwargs.get('dimensions')
        shape = kwargs.get('shape', kwargs.get('shape_global'))
        if grid is None:
            if shape is None:
                raise TypeError("Need either `grid` or `shape`")
        elif shape is None:
            if dimensions is not None and dimensions != (grid.dimensions or
                                                         subdomain.dimensions):
                raise TypeError("Need `shape` as not all `dimensions` are in `grid`")
            # TODO: Fix this:
            if subdomain:
                shape = subdomain.shape_local
            else:
                shape = grid.shape_local
        elif dimensions is None:
            raise TypeError("`dimensions` required if both `grid` and "
                            "`shape` are provided")
        else:
            # Got `grid`, `dimensions`, and `shape`. We sanity-check that the
            # Dimensions in `dimensions` also appearing in `grid` have same size
            # (given by `shape`) as that provided in `grid`
            if len(shape) != len(dimensions):
                raise ValueError("`shape` and `dimensions` must have the "
                                 "same number of entries")
            loc_shape = []
            for d, s in zip(dimensions, shape):
                if d in grid.dimensions:
                    size = grid.dimension_map[d]
                    if size.glb != s and s is not None:
                        raise ValueError("Dimension `%s` is given size `%d`, "
                                         "while `grid` says `%s` has size `%d` "
                                         % (d, s, d, size.glb))
                    else:
                        loc_shape.append(size.loc)
                else:
                    loc_shape.append(s)
            shape = tuple(loc_shape)
        return shape

    def __halo_setup__(self, **kwargs):
        halo = kwargs.get('halo')
        if halo is not None:
            return halo
        else:
            space_order = kwargs.get('space_order', 1)
            if isinstance(space_order, int):
                halo = (space_order, space_order)
            elif isinstance(space_order, tuple) and len(space_order) == 3:
                _, left_points, right_points = space_order
                halo = (left_points, right_points)
            else:
                raise TypeError("`space_order` must be int or 3-tuple of ints")
            return tuple(halo if i.is_Space else (0, 0) for i in self.dimensions)

    def __padding_setup__(self, **kwargs):
        padding = kwargs.get('padding')
        if padding is None:
            if kwargs.get('autopadding', configuration['autopadding']):
                # Auto-padding
                # 0-padding in all Dimensions except in the Fastest Varying Dimension,
                # `fvd`, which is the innermost one
                padding = [(0, 0) for i in self.dimensions[:-1]]
                fvd = self.dimensions[-1]
                # Let UB be a function that rounds up a value `x` to the nearest
                # multiple of the SIMD vector length, `vl`
                vl = configuration['platform'].simd_items_per_reg(self.dtype)
                ub = lambda x: int(ceil(x / vl)) * vl
                # Given the HALO and DOMAIN sizes, the right-PADDING is such that:
                # * the `fvd` size is a multiple of `vl`
                # * it contains *at least* `vl` points
                # This way:
                # * all first grid points along the `fvd` will be cache-aligned
                # * there is enough room to round up the loop trip counts to maximize
                #   the effectiveness SIMD vectorization
                fvd_pad_size = (ub(self._size_nopad[fvd]) - self._size_nopad[fvd]) + vl
                padding.append((0, fvd_pad_size))
                return tuple(padding)
            else:
                return tuple((0, 0) for d in self.dimensions)
        elif isinstance(padding, int):
            return tuple((0, padding) if d.is_Space else (0, 0) for d in self.dimensions)
        elif isinstance(padding, tuple) and len(padding) == self.ndim:
            return tuple((0, i) if isinstance(i, int) else i for i in padding)
        else:
            raise TypeError("`padding` must be int or %d-tuple of ints" % self.ndim)

    @property
    def space_order(self):
        """The space order."""
        return self._space_order

    def sum(self, p=None, dims=None):
        """
        Generate a symbolic expression computing the sum of ``p`` points
        along the spatial dimensions ``dims``.

        Parameters
        ----------
        p : int, optional
            The number of summands. Defaults to the halo size.
        dims : tuple of Dimension, optional
            The Dimensions along which the sum is computed. Defaults to
            ``self``'s spatial dimensions.
        """
        points = []
        for d in (as_tuple(dims) or self.space_dimensions):
            if p is None:
                lp = self._size_inhalo[d].left
                rp = self._size_inhalo[d].right
            else:
                lp = p // 2 + p % 2
                rp = p // 2
            indices = [d - i for i in range(lp, 0, -1)]
            indices.extend([d + i for i in range(rp)])
            points.extend([self.subs({d: i}) for i in indices])
        return sum(points)

    def avg(self, p=None, dims=None):
        """
        Generate a symbolic expression computing the average of ``p`` points
        along the spatial dimensions ``dims``.

        Parameters
        ----------
        p : int, optional
            The number of summands. Defaults to the halo size.
        dims : tuple of Dimension, optional
            The Dimensions along which the average is computed. Defaults to
            ``self``'s spatial dimensions.
        """
        tot = self.sum(p, dims)
        return tot / len(tot.args)

    # Pickling support
    _pickle_kwargs = DiscreteFunction._pickle_kwargs +\
        ['space_order', 'shape_global', 'dimensions']


class TimeFunction(Function):

    """
    Tensor symbol representing a discrete function in symbolic equations.

    A TimeFunction carries multi-dimensional data and provides operations to create
    finite-differences approximations, in both space and time.

    A TimeFunction encapsulates space- and time-varying data.

    Parameters
    ----------
    name : str
        Name of the symbol.
    grid : Grid, optional
        Carries shape, dimensions, and dtype of the Function. When grid is not
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
    time_order : int, optional
        Discretization order for time derivatives. Defaults to 1.
    shape : tuple of ints, optional
        Shape of the domain region in grid points. Only necessary if `grid` isn't given.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if `grid` isn't given.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to `np.float32`.
    save : int or Buffer, optional
        By default, ``save=None``, which indicates the use of alternating buffers. This
        enables cyclic writes to the TimeFunction. For example, if the TimeFunction
        ``u(t, x)`` has shape (3, 100), then, in an Operator, ``t`` will assume the
        values ``1, 2, 0, 1, 2, 0, 1, ...`` (note that the very first value depends
        on the stencil equation in which ``u`` is written.). The default size of the time
        buffer when ``save=None`` is ``time_order + 1``.  To specify a different size for
        the time buffer, one should use the syntax ``save=Buffer(mysize)``.
        Alternatively, if all of the intermediate results are required (or, simply, to
        avoid using an alternating buffer), an explicit value for ``save`` ( an integer)
        must be provided.
    time_dim : Dimension, optional
        TimeDimension to be used in the TimeFunction. Defaults to ``grid.time_dim``.
    staggered : Dimension or tuple of Dimension or Stagger, optional
        Define how the Function is staggered.
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

    Examples
    --------

    Creation

    >>> from devito import Grid, TimeFunction
    >>> grid = Grid(shape=(4, 4))
    >>> f = TimeFunction(name='f', grid=grid)
    >>> f
    f(t, x, y)
    >>> g = TimeFunction(name='g', grid=grid, time_order=2)
    >>> g
    g(t, x, y)

    First-order derivatives through centered finite-difference approximations

    >>> f.dx
    Derivative(f(t, x, y), x)
    >>> f.dt
    Derivative(f(t, x, y), t)
    >>> g.dt
    Derivative(g(t, x, y), t)

    When using the alternating buffer protocol, the size of the time dimension
    is given by ``time_order + 1``

    >>> f.shape
    (2, 4, 4)
    >>> g.shape
    (3, 4, 4)

    One can drop the alternating buffer protocol specifying a value for ``save``

    >>> h = TimeFunction(name='h', grid=grid, save=20)
    >>> h
    h(time, x, y)
    >>> h.shape
    (20, 4, 4)

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy uses
    ``*args`` to (re-)create the dimension arguments of the symbolic object.
    If the parameter ``grid`` is provided, the values for ``shape``,
    ``dimensions`` and ``dtype`` will be derived from it. When present, the
    parameter ``shape`` should only define the spatial shape of the grid. The
    temporal dimension will be inserted automatically as the leading dimension.
    """

    is_TimeFunction = True
    is_TimeDependent = True

    _time_position = 0
    """Position of time index among the function indices."""

    def __init_finalize__(self, *args, **kwargs):
        self.time_dim = kwargs.get('time_dim', self.dimensions[self._time_position])
        self._time_order = kwargs.get('time_order', 1)
        super(TimeFunction, self).__init_finalize__(*args, **kwargs)

        # Check we won't allocate too much memory for the system
        available_mem = virtual_memory().available
        if np.dtype(self.dtype).itemsize * self.size > available_mem:
            warning("Trying to allocate more memory for symbol %s " % self.name +
                    "than available on physical device, this will start swapping")
        if not isinstance(self.time_order, int):
            raise TypeError("`time_order` must be int")

        self.save = kwargs.get('save')

    @classmethod
    def __indices_setup__(cls, **kwargs):
        dimensions = kwargs.get('dimensions')
        staggered = kwargs.get('staggered')
        if dimensions is None:
            save = kwargs.get('save')
            grid = kwargs.get('grid')
            time_dim = kwargs.get('time_dim')

            if time_dim is None:
                time_dim = grid.time_dim if isinstance(save, int) else grid.stepping_dim
            elif not (isinstance(time_dim, Dimension) and time_dim.is_Time):
                raise TypeError("`time_dim` must be a time dimension")
            dimensions = list(Function.__indices_setup__(**kwargs)[0])
            dimensions.insert(cls._time_position, time_dim)
        return Function.__indices_setup__(dimensions=dimensions, staggered=staggered)

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        save = kwargs.get('save') or None  # Force to None if 0/False/None/...
        shape = kwargs.get('shape')
        time_order = kwargs.get('time_order', 1)

        if grid is None:
            if shape is None:
                raise TypeError("Need either `grid` or `shape`")
            if save is not None:
                raise TypeError("Ambiguity detected: provide either `grid` and `save` "
                                "or just `shape` ")
        elif shape is None:
            shape = list(grid.shape_local)
            if save is None:
                shape.insert(cls._time_position, time_order + 1)
            elif isinstance(save, Buffer):
                shape.insert(cls._time_position, save.val)
            elif isinstance(save, int):
                shape.insert(cls._time_position, save)
            else:
                raise TypeError("`save` can be None, int or Buffer, not %s" % type(save))
        return tuple(shape)

    @cached_property
    def _fd_priority(self):
        return super(TimeFunction, self)._fd_priority + .1

    @property
    def time_order(self):
        """The time order."""
        return self._time_order

    @property
    def forward(self):
        """Symbol for the time-forward state of the TimeFunction."""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.dimensions[self._time_position]

        return self._subs(_t, _t + i * _t.spacing)

    @property
    def backward(self):
        """Symbol for the time-backward state of the TimeFunction."""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.dimensions[self._time_position]

        return self._subs(_t, _t - i * _t.spacing)

    @property
    def _time_size(self):
        return self.shape_allocated[self._time_position]

    @property
    def _time_buffering(self):
        return not is_integer(self.save)

    @property
    def _time_buffering_default(self):
        return self._time_buffering and not isinstance(self.save, Buffer)

    def _arg_check(self, args, intervals):
        super(TimeFunction, self)._arg_check(args, intervals)
        key_time_size = args[self.name].shape[self._time_position]
        if self._time_buffering and self._time_size != key_time_size:
            raise InvalidArgument("Expected `time_size=%d` for runtime "
                                  "value `%s`, found `%d` instead"
                                  % (self._time_size, self.name, key_time_size))

    # Pickling support
    _pickle_kwargs = Function._pickle_kwargs + ['time_order', 'save', 'time_dim']


class SubFunction(Function):

    """
    A Function bound to a "parent" DiscreteFunction.

    A SubFunction hands control of argument binding and halo exchange to its
    parent DiscreteFunction.
    """

    def __init_finalize__(self, *args, **kwargs):
        super(SubFunction, self).__init_finalize__(*args, **kwargs)
        self._parent = kwargs['parent']

    def __padding_setup__(self, **kwargs):
        # SubFunctions aren't expected to be used in time-consuming loops
        return tuple((0, 0) for i in range(self.ndim))

    def _halo_exchange(self):
        return

    def _arg_values(self, **kwargs):
        if self.name in kwargs:
            raise RuntimeError("`%s` is a SubFunction, so it can't be assigned "
                               "a value dynamically" % self.name)
        else:
            return self._parent._arg_defaults(alias=self._parent).reduce_all()

    @property
    def parent(self):
        return self._parent

    _pickle_kwargs = Function._pickle_kwargs + ['parent']
