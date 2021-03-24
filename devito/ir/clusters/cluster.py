from itertools import chain

import numpy as np
from cached_property import cached_property

from devito.ir.equations import ClusterizedEq
from devito.ir.support import (PARALLEL, PARALLEL_IF_PVT, IterationSpace, DataSpace,
                               Scope, detect_io, normalize_properties)
from devito.symbolics import estimate_cost
from devito.tools import as_tuple, flatten, frozendict
from devito.types import normalize_syncs

__all__ = ["Cluster", "ClusterGroup"]


class Cluster(object):

    """
    A Cluster is an ordered sequence of expressions in an IterationSpace.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        An ordered sequence of expressions computing a tensor.
    ispace : IterationSpace
        The cluster iteration space.
    dspace : DataSpace
        The cluster data space.
    guards : dict, optional
        Mapper from Dimensions to expr-like, representing the conditions under
        which the Cluster should be computed.
    properties : dict, optional
        Mapper from Dimensions to Property, describing the Cluster properties
        such as its parallel Dimensions.
    syncs : dict, optional
        Mapper from Dimensions to lists of SyncOps, that is ordered sequences of
        synchronization operations that must be performed in order to compute the
        Cluster asynchronously.
    """

    def __init__(self, exprs, ispace, dspace, guards=None, properties=None, syncs=None):
        self._exprs = tuple(ClusterizedEq(i, ispace=ispace, dspace=dspace)
                            for i in as_tuple(exprs))
        self._ispace = ispace
        self._dspace = dspace
        self._guards = frozendict(guards or {})
        self._syncs = frozendict(syncs or {})

        properties = dict(properties or {})
        properties.update({i.dim: properties.get(i.dim, set()) for i in ispace.intervals})
        self._properties = frozendict(properties)

    def __repr__(self):
        return "Cluster([%s])" % ('\n' + ' '*9).join('%s' % i for i in self.exprs)

    @classmethod
    def from_clusters(cls, *clusters):
        """
        Build a new Cluster from a sequence of pre-existing Clusters with
        compatible IterationSpace.
        """
        assert len(clusters) > 0
        root = clusters[0]
        if not all(root.ispace.is_compatible(c.ispace) for c in clusters):
            raise ValueError("Cannot build a Cluster from Clusters with "
                             "incompatible IterationSpace")
        if not all(root.guards == c.guards for c in clusters):
            raise ValueError("Cannot build a Cluster from Clusters with "
                             "non-homogeneous guards")

        exprs = chain(*[c.exprs for c in clusters])
        ispace = IterationSpace.union(*[c.ispace for c in clusters])
        dspace = DataSpace.union(*[c.dspace for c in clusters])

        guards = root.guards

        properties = {}
        for c in clusters:
            for d, v in c.properties.items():
                properties[d] = normalize_properties(properties.get(d, v), v)

        try:
            syncs = normalize_syncs(*[c.syncs for c in clusters])
        except ValueError:
            raise ValueError("Cannot build a Cluster from Clusters with "
                             "non-compatible synchronization operations")

        return Cluster(exprs, ispace, dspace, guards, properties, syncs)

    def rebuild(self, *args, **kwargs):
        """
        Build a new Cluster from the attributes given as keywords. All other
        attributes are taken from ``self``.
        """
        # Shortcut for backwards compatibility
        if args:
            if len(args) != 1:
                raise ValueError("rebuild takes at most one positional argument (exprs)")
            if kwargs.get('exprs'):
                raise ValueError("`exprs` provided both as arg and kwarg")
            kwargs['exprs'] = args[0]

        return Cluster(exprs=kwargs.get('exprs', self.exprs),
                       ispace=kwargs.get('ispace', self.ispace),
                       dspace=kwargs.get('dspace', self.dspace),
                       guards=kwargs.get('guards', self.guards),
                       properties=kwargs.get('properties', self.properties),
                       syncs=kwargs.get('syncs', self.syncs))

    @property
    def exprs(self):
        return self._exprs

    @property
    def ispace(self):
        return self._ispace

    @property
    def itintervals(self):
        return self.ispace.itintervals

    @property
    def sub_iterators(self):
        return self.ispace.sub_iterators

    @property
    def directions(self):
        return self.ispace.directions

    @property
    def dspace(self):
        return self._dspace

    @property
    def guards(self):
        return self._guards

    @property
    def properties(self):
        return self._properties

    @property
    def syncs(self):
        return self._syncs

    @cached_property
    def sync_locks(self):
        return frozendict({k: tuple(i for i in v if i.is_SyncLock)
                           for k, v in self.syncs.items()})

    @cached_property
    def free_symbols(self):
        return set().union(*[e.free_symbols for e in self.exprs])

    @cached_property
    def dimensions(self):
        return set().union(*[i._defines for i in self.ispace.dimensions])

    @cached_property
    def used_dimensions(self):
        """
        The Dimensions that *actually* appear among the expressions in ``self``.
        These do not necessarily coincide the IterationSpace Dimensions; for
        example, reduction or redundant (i.e., invariant) Dimensions won't
        appear in an expression.
        """
        return {i for i in self.free_symbols if i.is_Dimension}

    @cached_property
    def scope(self):
        return Scope(self.exprs)

    @cached_property
    def functions(self):
        return self.scope.functions

    @cached_property
    def has_increments(self):
        return any(e.is_Increment for e in self.exprs)

    @cached_property
    def is_scalar(self):
        return not any(f.is_Function for f in self.scope.writes)

    @cached_property
    def is_dense(self):
        """
        A Cluster is dense if at least one of the following conditions is True:

            * It is defined over a unique Grid and all of the Grid Dimensions
              are PARALLEL.
            * Only DiscreteFunctions are written and only affine index functions
              are used (e.g., `a[x+1, y-2]` is OK, while `a[b[x], y-2]` is not)
        """
        # Hopefully it's got a unique Grid and all Dimensions are PARALLEL (or
        # at most PARALLEL_IF_PVT). This is a quick and easy check so we try it first
        try:
            pset = {PARALLEL, PARALLEL_IF_PVT}
            grid = self.grid
            for d in grid.dimensions:
                if not any(pset & v for k, v in self.properties.items()
                           if d in k._defines):
                    raise ValueError
            return True
        except ValueError:
            pass

        # Fallback to legacy is_dense checks
        return (not any(e.conditionals for e in self.exprs) and
                not any(f.is_SparseFunction for f in self.functions) and
                not self.is_scalar and
                all(a.is_regular for a in self.scope.accesses))

    @cached_property
    def grid(self):
        if len(self.grids) == 1:
            return self.grids[0]
        raise ValueError("Cluster has no unique Grid")

    @cached_property
    def grids(self):
        """
        The Grid's over which the Cluster is defined.
        """
        return tuple(set(i.grid for i in self.exprs if i.grid is not None))

    @cached_property
    def dtype(self):
        """
        The arithmetic data type of the Cluster. If the Cluster performs
        floating point arithmetic, then the expressions performing integer
        arithmetic are ignored, assuming that they are only carrying out array
        index calculations. If two expressions perform floating point
        calculations with mixed precision, the data type with highest precision
        is returned.
        """
        dtypes = {i.dtype for i in self.exprs}
        fdtypes = {i for i in dtypes if np.issubdtype(i, np.floating)}
        if len(fdtypes) > 1:
            raise NotImplementedError("Unsupported Cluster with mixed floating "
                                      "point arithmetic %s" % str(fdtypes))
        elif len(fdtypes) == 1:
            return fdtypes.pop()
        elif len(dtypes) == 1:
            return dtypes.pop()
        else:
            raise ValueError("Unsupported Cluster [mixed integer arithmetic ?]")

    @cached_property
    def ops(self):
        """Number of operations performed at each iteration."""
        return sum(estimate_cost(i) for i in self.exprs)

    @cached_property
    def traffic(self):
        """
        The Cluster compulsary traffic (number of reads/writes), as a mapper
        from Functions to IntervalGroups.

        Notes
        -----
        If a Function is both read and written, then it is counted twice.
        """
        reads, writes = detect_io(self.exprs, relax=True)
        accesses = [(i, 'r') for i in reads] + [(i, 'w') for i in writes]
        ret = {}
        for i, mode in accesses:
            if not i.is_Tensor:
                continue
            elif i in self.dspace.parts:
                # Stencils extend the data spaces beyond the iteration spaces
                intervals = self.dspace.parts[i]
                # Assume that invariant dimensions always cause new loads/stores
                invariants = self.ispace.intervals.drop(intervals.dimensions)
                intervals = intervals.generate('union', invariants, intervals)
                ret[(i, mode)] = intervals
            else:
                ret[(i, mode)] = self.ispace.intervals
        return ret


class ClusterGroup(tuple):

    """
    An immutable, totally-ordered sequence of Clusters.

    Parameters
    ----------
    clusters : tuple of Clusters
        Input elements.
    itintervals : tuple of IterationIntervals, optional
        The region of iteration space shared by the ``clusters``.
    """

    def __new__(cls, clusters, itintervals=None):
        obj = super(ClusterGroup, cls).__new__(cls, flatten(as_tuple(clusters)))
        obj._itintervals = itintervals
        return obj

    @classmethod
    def concatenate(cls, *cgroups):
        return list(chain(*cgroups))

    @cached_property
    def exprs(self):
        return flatten(c.exprs for c in self)

    @cached_property
    def scope(self):
        return Scope(exprs=self.exprs)

    @cached_property
    def itintervals(self):
        """The prefix IterationIntervals common to all Clusters in self."""
        return self._itintervals

    @cached_property
    def guards(self):
        """The guards of each Cluster in self."""
        return tuple(i.guards for i in self)

    @cached_property
    def sync_locks(self):
        """The synchronization locks of each Cluster in self."""
        return tuple(i.sync_locks for i in self)

    @cached_property
    def dspace(self):
        """Return the DataSpace of this ClusterGroup."""
        return DataSpace.union(*[i.dspace.reset() for i in self])

    @cached_property
    def dtype(self):
        """
        The arithmetic data type of this ClusterGroup. If at least one
        Cluster performs floating point arithmetic, then Clusters performing
        integer arithmetic are ignored. If two Clusters perform floating
        point calculations with different precision, return the data type with
        highest precision.
        """
        dtypes = {i.dtype for i in self}
        fdtypes = {i for i in dtypes if np.issubdtype(i, np.floating)}
        if len(fdtypes) > 1:
            raise NotImplementedError("Unsupported ClusterGroup with mixed floating "
                                      "point arithmetic %s" % str(fdtypes))
        elif len(fdtypes) == 1:
            return fdtypes.pop()
        elif len(dtypes) == 1:
            return dtypes.pop()
        else:
            raise ValueError("Unsupported ClusterGroup [mixed integer arithmetic ?]")

    @cached_property
    def meta(self):
        """
        Returns
        -------
        dtype, DSpace
            The data type and the data space of the ClusterGroup.
        """
        return (self.dtype, self.dspace)
