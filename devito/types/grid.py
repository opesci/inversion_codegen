from collections import namedtuple

import numpy as np
from sympy import prod
from math import floor

from devito.data import LEFT, RIGHT
from devito.mpi import Distributor
from devito.tools import ReducerMap, as_tuple, memoized_meth
from devito.types.args import ArgProvider
from devito.types.constant import Constant
from devito.types.dense import Function
from devito.types.dimension import (Dimension, SpaceDimension, TimeDimension,
                                    SteppingDimension, SubDimension)
from devito.types.equation import Eq

__all__ = ['Grid', 'SubDomain', 'SubDomainSet']


GlobalLocal = namedtuple('GlobalLocal', 'glb loc')


class Grid(ArgProvider):

    """
    A cartesian grid that encapsulates a computational domain over which
    to discretize a Function.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the computational domain in grid points.
    extent : tuple of floats, optional
        Physical extent of the domain in m; defaults to a unit box of extent 1m
        in all dimensions.
    origin : tuple of floats, optional
        Physical coordinate of the origin of the domain; defaults to 0.0 in all
        dimensions.
    dimensions : tuple of SpaceDimension, optional
        The dimensions of the computational domain encapsulated by this Grid.
    time_dimension : TimeDimension, optional
        The dimension used to define time in a `TimeFunction` created from
        this Grid.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type, used as default
        data type to be inherited by all Functions created from this Grid.
        Defaults to ``np.float32``.
    subdomains : tuple of SubDomain, optional
        If no subdomains are specified, the Grid only defines the two default
        subdomains ``interior`` and ``domain``.
    comm : MPI communicator, optional
        The set of processes over which the grid is distributed. Only relevant in
        case of MPI execution.

    Examples
    --------
    >>> from devito import Grid, Function
    >>> grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
    >>> f = Function(name='f', grid=grid)
    >>> f.shape
    (4, 4)
    >>> f.dimensions
    (x, y)
    >>> f.dtype
    <class 'numpy.float32'>

    In a Function, the domain defined by a Grid is often surrounded by a "halo
    region", which guarantees the correctness of stencil updates nearby the
    domain boundary. However, the size of the halo region does *not* depend on
    the Grid; for more information, refer to ``Function.__doc__``.

    >>> f.shape_with_halo
    (6, 6)

    Notes
    -----
    A Grid encapsulates the topology and geometry information of the
    computational domain that a Function can be discretized on.  As such, it
    defines and provides the physical coordinate information of the logical
    cartesian grid underlying the discretized Functions.  For example, the
    conventions for defining the coordinate space in 2D are:

        .. code-block:: python

                       x
             |----------------------->
             |  origin
             |     o------------o
             |     |            |
             |     |            |
             |     |   DOMAIN   | extent[1]
         y   |     |            |
             |     |            |
             |     |  extent[0] |
             |     o------------o
             |             origin + extent
             |
             v
    """

    _default_dimensions = ('x', 'y', 'z')

    def __init__(self, shape, extent=None, origin=None, dimensions=None,
                 time_dimension=None, dtype=np.float32, subdomains=None,
                 comm=None):
        self._shape = as_tuple(shape)
        self._extent = as_tuple(extent or tuple(1. for _ in self.shape))
        self._dtype = dtype

        if dimensions is None:
            # Create the spatial dimensions and constant spacing symbols
            assert(self.dim <= 3)
            dim_names = self._default_dimensions[:self.dim]
            dim_spacing = tuple(self._const(name='h_%s' % n, value=v, dtype=self.dtype)
                                for n, v in zip(dim_names, self.spacing))
            self._dimensions = tuple(SpaceDimension(name=n, spacing=s)
                                     for n, s in zip(dim_names, dim_spacing))
        else:
            self._dimensions = dimensions

        self._distributor = Distributor(self.shape, self.dimensions, comm)

        # Initialize SubDomains
        subdomains = tuple(i for i in (Domain(), Interior(), *as_tuple(subdomains)))
        for counter, i in enumerate(subdomains):
            i.__subdomain_finalize__(self.dimensions, self.shape,
                                     distributor=self._distributor, counter=counter)
        self._subdomains = subdomains

        origin = as_tuple(origin or tuple(0. for _ in self.shape))
        self._origin = tuple(self._const(name='o_%s' % d.name, value=v, dtype=self.dtype)
                             for d, v in zip(self.dimensions, origin))

        # Sanity check
        assert (self.dim == len(self.origin) == len(self.extent) == len(self.spacing))

        # Store or create default symbols for time and stepping dimensions
        if time_dimension is None:
            spacing = self._const(name='dt', dtype=self.dtype)
            self._time_dim = self._make_time_dim(spacing)
            self._stepping_dim = self._make_stepping_dim(self.time_dim, name='t')
        elif isinstance(time_dimension, TimeDimension):
            self._time_dim = time_dimension
            self._stepping_dim = self._make_stepping_dim(self.time_dim)
        else:
            raise ValueError("`time_dimension` must be None or of type TimeDimension")

    def __repr__(self):
        return "Grid[extent=%s, shape=%s, dimensions=%s]" % (
            self.extent, self.shape, self.dimensions
        )

    @property
    def extent(self):
        """Physical extent of the domain in m."""
        return self._extent

    @property
    def dtype(self):
        """Data type inherited by all Functions defined on this Grid."""
        return self._dtype

    @property
    def origin(self):
        """Physical coordinates of the domain origin."""
        return self._origin

    @property
    def dimensions(self):
        """Spatial dimensions of the computational domain."""
        return self._dimensions

    @property
    def dim(self):
        """Problem dimension, or number of spatial dimensions."""
        return len(self.shape)

    @property
    def time_dim(self):
        """Time dimension associated with this Grid."""
        return self._time_dim

    @property
    def stepping_dim(self):
        """Stepping dimension associated with this Grid."""
        return self._stepping_dim

    @property
    def subdomains(self):
        """The SubDomains defined in this Grid."""
        return {i.name: i for i in self._subdomains}

    @property
    def interior(self):
        """The interior SubDomain of the Grid."""
        return self.subdomains['interior']

    @property
    def volume_cell(self):
        """Volume of a single cell e.g  h_x*h_y*h_z in 3D."""
        return prod(d.spacing for d in self.dimensions).subs(self.spacing_map)

    @property
    def spacing(self):
        """Spacing between grid points in m."""
        spacing = (np.array(self.extent) / (np.array(self.shape) - 1)).astype(self.dtype)
        return as_tuple(spacing)

    @property
    def spacing_symbols(self):
        """Symbols representing the grid spacing in each SpaceDimension"""
        return as_tuple(d.spacing for d in self.dimensions)

    @property
    def spacing_map(self):
        """Map between spacing symbols and their values for each SpaceDimension."""
        return dict(zip(self.spacing_symbols, self.spacing))

    @property
    def origin_offset(self):
        """Offset of the local (per-process) origin from the domain origin."""
        grid_origin = [min(i) for i in self.distributor.glb_numb]
        assert len(grid_origin) == len(self.spacing)
        return tuple(i*h for i, h in zip(grid_origin, self.spacing))

    @property
    def shape(self):
        """Shape of the physical domain."""
        return self._shape

    @property
    def shape_local(self):
        """Shape of the local (per-process) physical domain."""
        return self._distributor.shape

    @property
    def dimension_map(self):
        """Map between SpaceDimensions and their global/local size."""
        return {d: GlobalLocal(g, l)
                for d, g, l in zip(self.dimensions, self.shape, self.shape_local)}

    @property
    def distributor(self):
        """The Distributor used for domain decomposition."""
        return self._distributor

    @property
    def comm(self):
        """The MPI communicator used for domain decomposition."""
        return self._distributor.comm

    def is_distributed(self, dim):
        """True if ``dim`` is a distributed Dimension, False otherwise."""
        return any(dim is d for d in self.distributor.dimensions)

    @property
    def _const(self):
        """The type to be used to create constant symbols."""
        return Constant

    def _make_stepping_dim(self, time_dim, name=None):
        """Create a SteppingDimension for this Grid."""
        if name is None:
            name = '%s_s' % time_dim.name
        return SteppingDimension(name=name, parent=time_dim)

    def _make_time_dim(self, spacing):
        """Create a TimeDimension for this Grid."""
        return TimeDimension(name='time', spacing=spacing)

    @memoized_meth
    def _arg_defaults(self):
        """A map of default argument values defined by this Grid."""
        args = ReducerMap()

        for k, v in self.dimension_map.items():
            args.update(k._arg_defaults(_min=0, size=v.loc))

        if self.distributor.is_parallel:
            distributor = self.distributor
            args[distributor._obj_comm.name] = distributor._obj_comm.value
            args[distributor._obj_neighborhood.name] = distributor._obj_neighborhood.value

        return args

    def __getstate__(self):
        state = self.__dict__.copy()
        # A Distributor wraps an MPI communicator, which can't and shouldn't be pickled
        state.pop('_distributor')
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._distributor = Distributor(self.shape, self.dimensions)


class SubDomain(object):

    """
    Base class to define Grid subdomains.

    To create a new SubDomain, all one needs to do is overriding :meth:`define`.
    This method takes as input a set of Dimensions and produce a mapper

        ``M : Dimensions -> {d, ('left', N), ('middle', N, M), ('right', N)}``

    so that:

        * If ``M(d) = d``, then the SubDomain spans the entire Dimension ``d``.
        * If ``M(d) = ('left', N)``, then the SubDomain spans a contiguous
          region of ``N`` points starting at ``d``\'s left extreme.
        * ``M(d) = ('right', N)`` is analogous to the case above.
        * If ``M(d) = ('middle', N, M)``, then the SubDomain spans a contiguous
          region of ``d_size - (N + M)`` points starting at ``N`` and finishing
          at ``d_sizeM - M``.

    Examples
    --------
    An "Inner" SubDomain, which spans the entire domain except for an exterior
    boundary region of ``thickness=3``, can be implemented as follows

    >>> from devito import SubDomain
    >>> class Inner(SubDomain):
    ...     name = 'inner'
    ...     def define(self, dimensions):
    ...         return {d: ('middle', 3, 3) for d in dimensions}

    Like before, but now spanning the entire ``y`` Dimension of a three-dimensional
    grid

    >>> class InnerY(SubDomain):
    ...     name = 'inner_y'
    ...     def define(self, dimensions):
    ...         x, y, z = dimensions
    ...         return {x: ('middle', 3, 3), y: y, z: ('middle', 3, 3)}

    See Also
    --------
    Domain : An example of preset SubDomain.
    Interior : An example of preset Subdomain.
    """

    name = None
    """A unique name for the SubDomain."""

    def __init__(self):
        if self.name is None:
            raise ValueError("SubDomain requires a `name`")
        self._dimensions = None

    def __subdomain_finalize__(self, dimensions, shape, distributor=None, **kwargs):
        # Create the SubDomain's SubDimensions
        sub_dimensions = []
        sdshape = []
        counter = kwargs.get('counter', 0) - 1
        for k, v, s in zip(self.define(dimensions).keys(),
                           self.define(dimensions).values(), shape):
            if isinstance(v, Dimension):
                sub_dimensions.append(v)
                sdshape.append(s)
            else:
                sd_name = 'i%d%s' % (counter, k.name)
                try:
                    # Case ('middle', int, int)
                    side, thickness_left, thickness_right = v
                    if side != 'middle':
                        raise ValueError("Expected side 'middle', not `%s`" % side)
                    sub_dimensions.append(SubDimension.middle(sd_name, k, thickness_left,
                                                              thickness_right))
                    sdshape.append(s-thickness_left-thickness_right)
                except ValueError:
                    side, thickness = v
                    if side == 'left':
                        if s-thickness < 0:
                            raise ValueError("Maximum thickness of dimension %s "
                                             "is %d, not %d" % (k.name, s, thickness))
                        sub_dimensions.append(SubDimension.left(sd_name, k,
                                                                thickness))
                        sdshape.append(thickness)
                    elif side == 'right':
                        if s-thickness < 0:
                            raise ValueError("Maximum thickness of dimension %s "
                                             "is %d, not %d" % (k.name, s, thickness))
                        sub_dimensions.append(SubDimension.right(sd_name, k,
                                                                 thickness))
                        sdshape.append(thickness)
                    else:
                        raise ValueError("Expected sides 'left|right', not `%s`" % side)
        self._dimensions = tuple(sub_dimensions)

        # Set the SubDomain shape
        self._shape = tuple(sdshape)

        # Derive the local shape for `SubDomain`'s on distributed grids along with the
        # memory access map for any `Function` defined on this `SubDomain`.
        access_map = {}
        shift = {}
        shape_local = []
        for dim, d, s in zip(sub_dimensions, distributor.decomposition, self._shape):
            if dim.is_Sub:
                c_name = 'c_%s' % dim.name
                shift[c_name] = Constant(name=c_name, dtype=np.int32)
                if dim._local:
                    if distributor and distributor.is_parallel:
                        if dim.thickness.right[1] == 0:
                            ls = len(d.loc_abs_numb)
                            minc = d.glb_min
                            maxc = d.glb_max - (ls-dim.thickness.left[1])
                            l = d.index_glb_to_loc(0, LEFT)
                            r = d.index_glb_to_loc(ls-dim.thickness.left[1], RIGHT)
                        else:
                            ls = len(d.loc_abs_numb)
                            minc = d.glb_min + (ls-dim.thickness.right[1])
                            maxc = d.glb_max
                            l = d.index_glb_to_loc(ls-dim.thickness.right[1], LEFT)
                            r = d.index_glb_to_loc(0, RIGHT)
                        if d.loc_abs_min > maxc:
                            shape_local.append(0)
                        elif d.loc_abs_max < minc:
                            shape_local.append(0)
                        else:
                            if l is None:
                                l = 0
                            if r is None:
                                r = 0
                            shape_local.append(ls-l-r)
                        shift[c_name].data = l
                        access_map.update({dim: dim-shift[c_name]})
                    else:
                        if dim.thickness.left[1] == 0:
                            shift[c_name].data = (s-dim.thickness.right[1])
                        else:
                            shift[c_name].data = 0
                        access_map.update({dim: dim-shift[c_name]})
                        shape_local.append(s)
                else:
                    if distributor and distributor.is_parallel:
                        tl = dim.thickness.left[1]
                        tr = dim.thickness.right[1]
                        ls = len(d.loc_abs_numb)
                        minc = d.glb_min + tl
                        maxc = d.glb_max - tr
                        l = d.index_glb_to_loc(tl, LEFT)
                        r = d.index_glb_to_loc(tr, RIGHT)
                        if d.loc_abs_min > maxc:
                            shape_local.append(0)
                        elif d.loc_abs_max < minc:
                            shape_local.append(0)
                        else:
                            if l is None:
                                l = 0
                            if r is None:
                                r = 0
                            shape_local.append(ls-l-r)
                        shift[c_name].data = l
                    else:
                        shift[c_name].data = dim.thickness.left[1]
                        shape_local.append(s)
                    access_map.update({dim: dim-shift[c_name]})
            else:
                shape_local.append(len(d.loc_abs_numb))
                access_map.update({dim: dim})
        self._access_map = access_map
        self._shape_local = tuple(shape_local)

    def __eq__(self, other):
        if not isinstance(other, SubDomain):
            return False
        return self.name == other.name and self.dimensions == other.dimensions

    def __hash__(self):
        return hash((self.name, self.dimensions))

    def __str__(self):
        return "SubDomain %s[%s]" % (self.name, self.dimensions)

    __repr__ = __str__

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def dimension_map(self):
        return {d.root: d for d in self.dimensions}

    @property
    def shape(self):
        return self._shape

    @property
    def shape_local(self):
        return self._shape_local

    def define(self, dimensions):
        """
        Parametrically describe the SubDomain w.r.t. a generic Grid.

        Notes
        -----
        This method should be overridden by each SubDomain subclass. For more
        information, refer to ``SubDomain.__doc__``.
        """
        raise NotImplementedError


class Domain(SubDomain):

    """
    The entire computational domain (== boundary + interior).
    """

    name = 'domain'

    def define(self, dimensions):
        return dict(zip(dimensions, dimensions))


class Interior(SubDomain):

    """
    The interior of the computational domain (i.e., boundaries are excluded).
    """

    name = 'interior'

    def define(self, dimensions):
        return {d: ('middle', 1, 1) for d in dimensions}


class SubDomainSet(SubDomain):
    """
    Class to define a set of N (a positive integer) subdomains.

    Parameters
    ----------
    **kwargs
        * N : int
            Number of subdomains.
        * bounds : tuple
            Tuple of numpy int32 arrays representing the bounds of
            each subdomain.

    Examples
    --------
    Set up an iterate upon a set of two subdomains:

    >>> import numpy as np
    >>> from devito import Grid, Function, Eq, Operator, SubDomainSet
    >>> Nx = 10
    >>> Ny = Nx
    >>> n_domains = 2

    Create a 'SubDomainSet object':

    >>> class MySubdomains(SubDomainSet):
    ...     name = 'mydomains'

    Set the bounds of the subdomains. The required format is:
    (xm, xM, ym, yM, ...) where xm is a vector specifying
    the number of grid points inwards from the 'left' boundary in the
    first grid dimension that each subdomain starts. xM is a vector
    specifying the number of grid points inwards from the 'right' of
    the domain in the first grid dimension that each subdomain ends.
    ym and yM are the equivalents for the second grid dimension.

    >>> xm = np.array([1, Nx/2+1], dtype=np.int32)
    >>> xM = np.array([Nx/2+1, 1], dtype=np.int32)

    Along a dimension where all bounds are the same we can use the
    following shorthand:

    >>> ym = 1 # which is equivalent to 'np.array([1, 1], dtype=np.int32)'
    >>> yM = 1

    Combine the data into the required form:

    >>> bounds = (xm, xM, ym, yM)

    Create our set of subdomains passing the number of domains and the
    bounds:

    >>> my_sd = MySubdomains(N=n_domains, bounds=bounds)

    Create a grid and iterate a function within the defined subdomains:

    >>> grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd, ))
    >>> f = Function(name='f', grid=grid, dtype=np.int32)
    >>> eq = Eq(f, f+1, subdomain=grid.subdomains['mydomains'])
    >>> op = Operator(eq)
    >>> summary = op.apply()
    >>> f.data
    Data([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
    """

    implicit_dimension = None
    """The implicit dimension of the SubDomainSet."""

    def __init__(self, **kwargs):
        super(SubDomainSet, self).__init__()
        if self.implicit_dimension is None:
            n = Dimension(name='n')
            self.implicit_dimension = n
        self._n_domains = kwargs.get('N', 1)
        self._global_bounds = kwargs.get('bounds', None)
        self._implicit_exprs = None

    def __subdomain_finalize__(self, dimensions, shape, distributor=None, **kwargs):
        # Create the SubDomain's SubDimensions
        sub_dimensions = []
        for d in dimensions:
            sub_dimensions.append(SubDimension.middle
                                  ('%si_%s' % (d.name, self.implicit_dimension.name),
                                   d, 0, 0))
        self._dimensions = tuple(sub_dimensions)

        # Compute the SubDomainSet shapes
        global_bounds = []
        for i in self._global_bounds:
            if isinstance(i, int):
                global_bounds.append(np.full(self._n_domains, i, dtype=np.int32))
            else:
                global_bounds.append(i)
        d_m = global_bounds[0::2]
        d_M = global_bounds[1::2]
        shapes = []
        for i in range(self._n_domains):
            dshape = []
            for s, m, M in zip(shape, d_m, d_M):
                assert(m.size == M.size)
                dshape.append(s-m[i]-M[i])
            shapes.append(as_tuple(dshape))
        self._shape = as_tuple(shapes)

        if distributor and distributor.is_parallel:
            # Now create local bounds based on distributor
            processed = []
            for dim, d, m, M in zip(dimensions, distributor.decomposition, d_m, d_M):
                bounds_m = np.zeros(m.shape, dtype=m.dtype)
                bounds_M = np.zeros(m.shape, dtype=m.dtype)
                for j in range(m.size):
                    lmin = d.glb_min + m[j]
                    lmax = d.glb_max - M[j]

                    # Check if the subdomain doesn't intersect with the decomposition
                    if lmin < d.loc_abs_min and lmax < d.loc_abs_min:
                        bounds_m[j] = d.loc_abs_max
                        bounds_M[j] = d.loc_abs_max
                        continue
                    if lmin > d.loc_abs_max and lmax > d.loc_abs_max:
                        bounds_m[j] = d.loc_abs_max
                        bounds_M[j] = d.loc_abs_max
                        continue

                    if lmin < d.loc_abs_min:
                        bounds_m[j] = 0
                    elif lmin > d.loc_abs_max:
                        bounds_m[j] = d.loc_abs_max
                        bounds_M[j] = d.loc_abs_max
                        continue
                    else:
                        bounds_m[j] = d.index_glb_to_loc(m[j], LEFT)

                    if lmax < d.loc_abs_min:
                        bounds_m[j] = d.loc_abs_max
                        bounds_M[j] = d.loc_abs_max
                        continue
                    elif lmax >= d.loc_abs_max:
                        bounds_M[j] = 0
                    else:
                        bounds_M[j] = d.index_glb_to_loc(M[j], RIGHT)

                processed.append(bounds_m)
                processed.append(bounds_M)
            self._local_bounds = as_tuple(processed)
        else:
            # Not distributed and hence local and global bounds are
            # equivalent.
            self._local_bounds = self._global_bounds

    @property
    def n_domains(self):
        return self._n_domains

    @property
    def bounds(self):
        return self._local_bounds

    def _create_implicit_exprs(self, grid):
        if not len(self._local_bounds) == 2*len(self.dimensions):
            raise ValueError("Left and right bounds must be supplied for each dimension")
        n_domains = self.n_domains
        i_dim = self.implicit_dimension
        dat = []
        # Organise the data contained in 'bounds' into a form such that the
        # associated implicit equations can easily be created.
        for j in range(len(self._local_bounds)):
            index = floor(j/2)
            d = self.dimensions[index]
            if j % 2 == 0:
                fname = d.min_name
            else:
                fname = d.max_name
            func = Function(name=fname, shape=(n_domains, ), dimensions=(i_dim, ),
                            grid=grid, dtype=np.int32)
            # Check if shorthand notation has been provided:
            if isinstance(self._local_bounds[j], int):
                bounds = np.full((n_domains,), self._local_bounds[j], dtype=np.int32)
                func.data[:] = bounds
            else:
                func.data[:] = self._local_bounds[j]
            dat.append(Eq(d.thickness[j % 2][0], func[i_dim]))
        return as_tuple(dat)
