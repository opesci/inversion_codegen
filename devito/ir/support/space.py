import abc
from collections import OrderedDict
from functools import reduce
from operator import mul

from cached_property import cached_property
from sympy import Expr

from devito.ir.support.vector import Vector, vmin, vmax
from devito.tools import (PartialOrderTuple, as_list, as_tuple, filter_ordered,
                          frozendict, toposort, is_integer)
from devito.types import Dimension, ModuloDimension

__all__ = ['NullInterval', 'Interval', 'IntervalGroup', 'IterationSpace', 'DataSpace',
           'Forward', 'Backward', 'Any']


class AbstractInterval(object):

    """
    An abstract representation of an iterated closed interval on Z.
    """

    __metaclass__ = abc.ABCMeta

    is_Null = False
    is_Defined = False

    def __init__(self, dim, stamp=0):
        self.dim = dim
        self.stamp = stamp

    def __eq__(self, o):
        return (type(self) == type(o) and
                self.dim is o.dim and
                self.stamp == o.stamp)

    is_compatible = __eq__

    def __hash__(self):
        return hash(self.dim.name)

    @abc.abstractmethod
    def _rebuild(self):
        return

    @abc.abstractproperty
    def relaxed(self):
        return

    def intersection(self, o):
        return self._rebuild()

    @abc.abstractmethod
    def union(self, o):
        return self._rebuild()

    def add(self, o):
        return self._rebuild()

    subtract = add

    def negate(self):
        return self._rebuild()

    zero = negate
    flip = negate
    lift = negate
    reset = negate
    switch = negate
    translate = negate


class NullInterval(AbstractInterval):

    """
    A degenerate iterated closed interval on Z.
    """

    is_Null = True

    def __repr__(self):
        return "%s[Null]<%d>" % (self.dim, self.stamp)

    def __hash__(self):
        return hash(self.dim)

    def _rebuild(self):
        return NullInterval(self.dim, self.stamp)

    @property
    def relaxed(self):
        return NullInterval(self.dim.root, self.stamp)

    def union(self, o):
        if self.dim is o.dim:
            return o._rebuild()
        else:
            raise ValueError("Cannot compute union of Intervals over "
                             "different Dimensions")

    def switch(self, d):
        return NullInterval(d, self.stamp)


class Interval(AbstractInterval):

    """
    Interval(dim, lower, upper)

    A concrete iterated closed interval on Z.

    An Interval defines the compact region

        [dim.symbolic_min + lower, dim.symbolic_max + upper]
    """

    is_Defined = True

    def __init__(self, dim, lower, upper, stamp=0):
        super(Interval, self).__init__(dim, stamp)

        try:
            self.lower = int(lower)
        except TypeError:
            assert isinstance(lower, Expr)
            self.lower = lower
        try:
            self.upper = int(upper)
        except TypeError:
            assert isinstance(upper, Expr)
            self.upper = upper

    def __repr__(self):
        return "%s[%s,%s]<%d>" % (self.dim, self.lower, self.upper, self.stamp)

    def __hash__(self):
        return hash((self.dim, self.offsets))

    def __eq__(self, o):
        if self is o:
            return True

        return (super(Interval, self).__eq__(o) and
                self.lower == o.lower and
                self.upper == o.upper)

    def _rebuild(self):
        return Interval(self.dim, self.lower, self.upper, self.stamp)

    @cached_property
    def size(self):
        """
        The Interval size, defined as the number of points iterated over through
        ``self.dim``, namely

            (dim.symbolic_max + upper - dim.symbolic_min - lower + 1) / dim.symbolic_incr

        Notes
        -----
        The Interval size is typically a function of several symbols (e.g.,
        `self.dim.symbolic_max`), and all such symbols must be mappable
        to actual numbers at `op.apply` time (i.e., the runtime values).
        When `self.dim` is an "unstructured Dimension", such as ModuloDimension,
        things can get nasty since the symbolic min/max/incr can literally be
        anything (any expression involving any Dimension/symbol/...), which makes
        it extremely complicated to numerically compute the size. However,
        the compiler only uses such unstructured Dimensions in well defined
        circumstances, which we explicitly handle here. Ultimately, therefore,
        we return a `size` that is made up of known symbols.
        """
        if self.dim.is_Custom and isinstance(self.dim.symbolic_min, ModuloDimension):
            # Special case 1)
            # Caused by the performance option `cire-rotate=True`
            d = self.dim.symbolic_min
            n = d.parent.symbolic_size

            # Iteration 0:
            assert is_integer(d.symbolic_min)
            assert is_integer(d.symbolic_incr)
            assert is_integer(self.dim.symbolic_max)
            assert self.lower == self.upper
            npoints = self.dim.symbolic_max - d.symbolic_min + 1
            # Iterations [1, ..., n-1]:
            assert d.symbolic_incr == self.dim.symbolic_max
            npoints += 1 * (n-1)
            npoints /= n
        else:
            # Typically we end up here (Dimension, SubDimension, IncrDimension)
            assert not self.dim.is_Modulo
            assert not self.dim.is_Conditional

            upper_extreme = self.dim.symbolic_max + self.upper
            lower_extreme = self.dim.symbolic_min + self.lower

            npoints = (upper_extreme - lower_extreme + 1)

        return npoints / self.dim.symbolic_incr

    @cached_property
    def min_size(self):
        return self.upper - self.lower + 1

    @property
    def relaxed(self):
        return Interval(self.dim.root, self.lower, self.upper, self.stamp)

    @property
    def offsets(self):
        return (self.lower, self.upper)

    def intersection(self, o):
        if self.is_compatible(o):
            svl, svu = Vector(self.lower, smart=True), Vector(self.upper, smart=True)
            ovl, ovu = Vector(o.lower, smart=True), Vector(o.upper, smart=True)
            return Interval(self.dim, vmax(svl, ovl)[0], vmin(svu, ovu)[0], self.stamp)
        else:
            return NullInterval(self.dim)

    def union(self, o):
        if o.is_Null and self.dim is o.dim:
            return self._rebuild()
        elif self.is_compatible(o):
            svl, svu = Vector(self.lower, smart=True), Vector(self.upper, smart=True)
            ovl, ovu = Vector(o.lower, smart=True), Vector(o.upper, smart=True)
            return Interval(self.dim, vmin(svl, ovl)[0], vmax(svu, ovu)[0], self.stamp)
        else:
            raise ValueError("Cannot compute union of non-compatible Intervals (%s, %s)" %
                             (self, o))

    def add(self, o):
        if not self.is_compatible(o):
            return self._rebuild()
        else:
            return Interval(self.dim, self.lower + o.lower, self.upper + o.upper,
                            self.stamp)

    def subtract(self, o):
        if not self.is_compatible(o):
            return self._rebuild()
        else:
            return Interval(self.dim, self.lower - o.lower, self.upper - o.upper,
                            self.stamp)

    def negate(self):
        return Interval(self.dim, -self.lower, -self.upper, self.stamp)

    def zero(self):
        return Interval(self.dim, 0, 0, self.stamp)

    def flip(self):
        return Interval(self.dim, self.upper, self.lower, self.stamp)

    def lift(self, v=None):
        if v is None:
            v = self.stamp + 1
        return Interval(self.dim, self.lower, self.upper, v)

    def reset(self):
        return Interval(self.dim, self.lower, self.upper, 0)

    def switch(self, d):
        return Interval(d, self.lower, self.upper, self.stamp)

    def translate(self, v):
        return Interval(self.dim, self.lower + v, self.upper + v, self.stamp)


class IntervalGroup(PartialOrderTuple):

    """
    A partially-ordered sequence of Intervals equipped with set-like
    operations.
    """

    @classmethod
    def reorder(cls, items, relations):
        if not all(isinstance(i, AbstractInterval) for i in items):
            raise ValueError("Cannot create an IntervalGroup from objects of type [%s]" %
                             ', '.join(str(type(i)) for i in items))
        # The relations are between dimensions, not intervals. So we take
        # care of that here
        ordering = filter_ordered(toposort(relations) + [i.dim for i in items])
        return sorted(items, key=lambda i: ordering.index(i.dim))

    def __eq__(self, o):
        # No need to look at the relations -- if the partial ordering is the same,
        # then the IntervalGroups are considered equal
        return len(self) == len(o) and all(i == j for i, j in zip(self, o))

    def __contains__(self, d):
        return any(i.dim is d for i in self)

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return "IntervalGroup[%s]" % (', '.join([repr(i) for i in self]))

    @cached_property
    def dimensions(self):
        return filter_ordered([i.dim for i in self])

    @property
    def size(self):
        if self:
            return reduce(mul, [i.size for i in self])
        else:
            return 0

    @cached_property
    def is_well_defined(self):
        """
        True if all Intervals are over different Dimensions,
        False otherwise.
        """
        return len(self.dimensions) == len(set(self.dimensions))

    @classmethod
    def generate(self, op, *interval_groups):
        """
        Create a new IntervalGroup from the iterative application of an
        operation to some IntervalGroups.

        Parameters
        ----------
        op : str
            Any legal Interval operation, such as 'intersection' or
            or 'union'.
        *interval_groups
            Input IntervalGroups.

        Examples
        --------
        >>> from devito import dimensions
        >>> x, y, z = dimensions('x y z')
        >>> ig0 = IntervalGroup([Interval(x, 1, -1)])
        >>> ig1 = IntervalGroup([Interval(x, 2, -2), Interval(y, 3, -3)])
        >>> ig2 = IntervalGroup([Interval(y, 2, -2), Interval(z, 1, -1)])

        >>> IntervalGroup.generate('intersection', ig0, ig1, ig2)
        IntervalGroup[x[2,-2]<0>, y[3,-3]<0>, z[1,-1]<0>]
        """
        mapper = {}
        for ig in interval_groups:
            for i in ig:
                mapper.setdefault(i.dim, []).append(i)
        intervals = []
        for v in mapper.values():
            # Create a new Interval through the concatenation v0.key(v1).key(v2)...
            interval = v[0]
            for i in v[1:]:
                interval = getattr(interval, op)(i)
            intervals.append(interval)
        relations = set().union(*[ig.relations for ig in interval_groups])
        return IntervalGroup(intervals, relations=relations)

    @cached_property
    def relaxed(self):
        return IntervalGroup.generate('union', IntervalGroup(i.relaxed for i in self))

    def is_compatible(self, o):
        """
        Two IntervalGroups are compatible iff they can be ordered according
        to some common partial ordering.
        """
        if set(self) != set(o):
            return False
        if all(i == j for i, j in zip(self, o)):
            # Same input ordering, definitely compatible
            return True
        try:
            self.add(o)
            return True
        except ValueError:
            # Cyclic dependence detected, there is no common partial ordering
            return False

    def _normalize(func):
        """
        A simple decorator to normalize the input of operator methods that
        expect an IntervalGroup as an operand.
        """
        def wrapper(self, o):
            if not isinstance(o, IntervalGroup):
                o = IntervalGroup(as_tuple(o))
            return func(self, o)
        return wrapper

    @_normalize
    def intersection(self, o):
        mapper = OrderedDict([(i.dim, i) for i in o])
        intervals = [i.intersection(mapper.get(i.dim, i)) for i in self]
        return IntervalGroup(intervals, relations=(self.relations | o.relations))

    @_normalize
    def add(self, o):
        mapper = OrderedDict([(i.dim, i) for i in o])
        intervals = [i.add(mapper.get(i.dim, NullInterval(i.dim))) for i in self]
        return IntervalGroup(intervals, relations=(self.relations | o.relations))

    @_normalize
    def subtract(self, o):
        mapper = OrderedDict([(i.dim, i) for i in o])
        intervals = [i.subtract(mapper.get(i.dim, NullInterval(i.dim))) for i in self]
        return IntervalGroup(intervals, relations=(self.relations | o.relations))

    def drop(self, d):
        # Dropping
        dims = set().union(*[i._defines for i in as_tuple(d)])
        intervals = [i._rebuild() for i in self if not i.dim._defines & dims]

        # Clean up relations
        relations = [tuple(i for i in r if i in intervals) for r in self.relations]

        return IntervalGroup(intervals, relations=relations)

    def negate(self):
        return IntervalGroup([i.negate() for i in self], relations=self.relations)

    def zero(self, d=None):
        d = self.dimensions if d is None else as_tuple(d)
        return IntervalGroup([i.zero() if i.dim in d else i for i in self],
                             relations=self.relations)

    def lift(self, d=None, v=None):
        d = set(self.dimensions if d is None else as_tuple(d))
        return IntervalGroup([i.lift(v) if i.dim._defines & d else i for i in self],
                             relations=self.relations)

    def reset(self):
        return IntervalGroup([i.reset() for i in self], relations=self.relations)

    def switch(self, d0, d1):
        return IntervalGroup([i.switch(d1) if i.dim is d0 else i for i in self],
                             relations=self.relations)

    def index(self, key):
        if isinstance(key, Interval):
            return super(IntervalGroup, self).index(key)
        elif isinstance(key, Dimension):
            return super(IntervalGroup, self).index(self[key])
        raise ValueError("Expected Interval or Dimension, got `%s`" % type(key))

    def __getitem__(self, key):
        if is_integer(key):
            return super(IntervalGroup, self).__getitem__(key)
        elif isinstance(key, slice):
            retval = super(IntervalGroup, self).__getitem__(key)
            return IntervalGroup(retval, relations=self.relations)

        if not self.is_well_defined:
            raise ValueError("Cannot fetch Interval from ill defined Space")

        if not isinstance(key, Dimension):
            return NullInterval(key)

        for i in self:
            if i.dim is key:
                return i
            if key.is_NonlinearDerived and i.dim in key._defines:
                # NonlinearDerived Dimensions cannot appear in iteration Intervals,
                # but their parent can
                return i

        return NullInterval(key)


class IterationDirection(object):

    """
    A representation of the direction in which an iteration space is traversed.
    """

    def __init__(self, name):
        self._name = name

    def __eq__(self, other):
        return isinstance(other, IterationDirection) and self._name == other._name

    def __repr__(self):
        return self._name

    def __hash__(self):
        return hash(self._name)


Forward = IterationDirection('++')
"""Forward iteration direction ('++')."""

Backward = IterationDirection('--')
"""Backward iteration direction ('--')."""

Any = IterationDirection('*')
"""Wildcard direction (both '++' and '--' would be OK)."""


class IterationInterval(object):

    """
    An Interval associated with an IterationDirection.
    """

    def __init__(self, interval, direction, sub_iterators):
        self.interval = interval
        self.direction = direction
        self.sub_iterators = sub_iterators

    def __repr__(self):
        return "%s%s" % (self.interval, self.direction)

    def __eq__(self, other):
        return self.direction is other.direction and self.interval == other.interval

    def __hash__(self):
        return hash((self.interval, self.direction))

    @property
    def dim(self):
        return self.interval.dim

    @property
    def offsets(self):
        return self.interval.offsets

    @property
    def size(self):
        return self.interval.size


class Space(object):

    """
    A compact N-dimensional space defined by N Intervals.

    Parameters
    ----------
    intervals : tuple of Intervals
        Space description.
    """

    def __init__(self, intervals):
        if isinstance(intervals, IntervalGroup):
            self._intervals = intervals
        else:
            self._intervals = IntervalGroup(as_tuple(intervals))

    def __repr__(self):
        return "%s[%s]" % (self.__class__.__name__,
                           ", ".join(repr(i) for i in self.intervals))

    def __eq__(self, other):
        return isinstance(other, Space) and self.intervals == other.intervals

    def __hash__(self):
        return hash(self.intervals)

    def __len__(self):
        return len(self.intervals)

    def __iter__(self):
        for i in self.intervals:
            yield i

    @property
    def intervals(self):
        return self._intervals

    @cached_property
    def dimensions(self):
        return filter_ordered(self.intervals.dimensions)

    @property
    def size(self):
        return self.intervals.size

    @property
    def dimension_map(self):
        """
        Map between the Space Dimensions and the size of their iterated region.
        """
        return OrderedDict([(i.dim, i.size) for i in self.intervals])


class DataSpace(Space):

    """
    Represent a data space as an enriched Space.

    Parameters
    ----------
    intervals : tuple of Intervals
        Data space description.
    parts : dict
        A mapper from Functions to IntervalGroup, describing the individual
        components of the data space.
    """

    def __init__(self, intervals, parts):
        super(DataSpace, self).__init__(intervals)
        self._parts = frozendict(parts)

    def __eq__(self, other):
        return isinstance(other, DataSpace) and\
            self.intervals == other.intervals and self.parts == other.parts

    def __hash__(self):
        return hash((super(DataSpace, self).__hash__(), self.parts))

    @classmethod
    def union(cls, *others):
        if not others:
            return DataSpace(IntervalGroup(), {})
        intervals = IntervalGroup.generate('union', *[i.intervals for i in others])
        parts = {}
        for i in others:
            for k, v in i.parts.items():
                parts.setdefault(k, []).append(v)
        parts = {k: IntervalGroup.generate('union', *v) for k, v in parts.items()}
        return DataSpace(intervals, parts)

    @property
    def parts(self):
        return self._parts

    @cached_property
    def relaxed(self):
        """
        A view of the DataSpace assuming that any SubDimensions entirely span
        their root Dimension.
        """
        return DataSpace(self.intervals.relaxed,
                         {k: v.relaxed for k, v in self.parts.items()})

    def __getitem__(self, key):
        ret = self.intervals[key]
        if ret.is_Null:
            try:
                ret = self._parts[key]
            except KeyError:
                ret = IntervalGroup()
        return ret

    def zero(self, d=None):
        intervals = self.intervals.zero(d)
        parts = {k: v.zero(d) for k, v in self.parts.items()}
        return DataSpace(intervals, parts)

    def lift(self, d=None):
        intervals = self.intervals.lift(d)
        parts = {k: v.lift(d) for k, v in self.parts.items()}
        return DataSpace(intervals, parts)

    def reset(self):
        intervals = self.intervals.reset()
        parts = {k: v.reset() for k, v in self.parts.items()}
        return DataSpace(intervals, parts)

    def project(self, cond):
        """
        Create a new DataSpace in which only some of the Dimensions in
        ``self`` are retained. In particular, a dimension ``d`` in ``self``
        is retained if:

            * either ``cond(d)`` is True (``cond`` is a callable),
            * or ``d in cond`` is True (``cond`` is an iterable)
        """
        if callable(cond):
            func = cond
        else:
            func = lambda i: i in cond
        intervals = [i for i in self.intervals if func(i.dim)]
        return DataSpace(intervals, self.parts)


class IterationSpace(Space):

    """
    Represent an iteration space as an enriched Space.

    Parameters
    ----------
    intervals : IntervalGroup
        Iteration space description.
    sub_iterators : dict, optional
        A mapper from Dimensions in ``intervals`` to iterables of
        DerivedDimensions defining sub-regions of iteration.
    directions : dict, optional
        A mapper from Dimensions in ``intervals`` to IterationDirections.
    """

    def __init__(self, intervals, sub_iterators=None, directions=None):
        super(IterationSpace, self).__init__(intervals)

        # Normalize sub-iterators
        sub_iterators = sub_iterators or {}
        self._sub_iterators = frozendict([(k, tuple(filter_ordered(as_tuple(v))))
                                          for k, v in sub_iterators.items()])

        # Normalize directions
        if directions is None:
            self._directions = frozendict([(i.dim, Any) for i in self.intervals])
        else:
            self._directions = frozendict(directions)

    def __repr__(self):
        ret = ', '.join(["%s%s" % (repr(i), repr(self.directions[i.dim]))
                         for i in self.intervals])
        return "IterationSpace[%s]" % ret

    def __eq__(self, other):
        if self is other:
            return True

        return (isinstance(other, IterationSpace) and
                self.intervals == other.intervals and
                self.directions == other.directions)

    def __lt__(self, other):
        """
        A rudimentary less-then comparison between two IterationSpaces.
        """
        return len(self.itintervals) < len(other.itintervals)

    def __hash__(self):
        return hash((super(IterationSpace, self).__hash__(), self.sub_iterators,
                     self.directions))

    def __getitem__(self, key):
        return self.intervals[key]

    @classmethod
    def union(cls, *others):
        if not others:
            return IterationSpace(IntervalGroup())
        elif len(others) == 1:
            return others[0]

        intervals = IntervalGroup.generate('union', *[i.intervals for i in others])

        directions = {}
        for i in others:
            for k, v in i.directions.items():
                if directions.get(k, Any) in (Any, v):
                    # No direction yet, or Any, or simply identical to /v/
                    directions[k] = v
                elif v is not Any:
                    # Clash detected
                    raise ValueError("Cannot compute the union of `IterationSpace`s "
                                     "with incompatible directions")

        sub_iterators = {}
        for i in others:
            for k, v in i.sub_iterators.items():
                ret = sub_iterators.setdefault(k, [])
                ret.extend([d for d in v if d not in ret])

        return IterationSpace(intervals, sub_iterators, directions)

    def add(self, other):
        return IterationSpace(self.intervals.add(other), self.sub_iterators,
                              self.directions)

    def augment(self, sub_iterators):
        """
        Create a new IterationSpace with additional sub-iterators.
        """
        items = dict(self.sub_iterators)
        for k, v in sub_iterators.items():
            if k not in self.intervals:
                continue
            items[k] = as_list(items.get(k))
            for i in as_list(v):
                if i not in items[k]:
                    items[k].append(i)

        return IterationSpace(self.intervals, items, self.directions)

    def reset(self):
        return IterationSpace(self.intervals.reset(), self.sub_iterators, self.directions)

    def project(self, cond):
        """
        Create a new IterationSpace in which only some Dimensions
        in ``self`` are retained. In particular, a Dimension ``d`` in ``self`` is
        retained if:

            * either ``cond(d)`` is true (``cond`` is a callable),
            * or ``d in cond`` is true (``cond`` is an iterable)
        """
        if callable(cond):
            func = cond
        else:
            func = lambda i: i in cond

        intervals = [i for i in self.intervals if func(i.dim)]
        sub_iterators = {k: v for k, v in self.sub_iterators.items() if func(k)}
        directions = {k: v for k, v in self.directions.items() if func(k)}
        return IterationSpace(intervals, sub_iterators, directions)

    def zero(self, d=None):
        intervals = self.intervals.zero(d)

        return IterationSpace(intervals, self.sub_iterators, self.directions)

    def lift(self, d=None, v=None):
        intervals = self.intervals.lift(d, v)

        return IterationSpace(intervals, self.sub_iterators, self.directions)

    def is_compatible(self, other):
        """
        A relaxed version of ``__eq__``, in which only non-derived dimensions
        are compared for equality.
        """
        return (self.intervals.is_compatible(other.intervals) and
                self.nonderived_directions == other.nonderived_directions)

    def is_forward(self, dim):
        return self.directions[dim] is Forward

    @property
    def itdimensions(self):
        return self.intervals.dimensions

    @property
    def relations(self):
        return self.intervals.relations

    @property
    def sub_iterators(self):
        return self._sub_iterators

    @property
    def directions(self):
        return self._directions

    @cached_property
    def itintervals(self):
        return tuple(IterationInterval(
            i, self.directions[i.dim], self.sub_iterators.get(i.dim)
        ) for i in self.intervals)

    @cached_property
    def dimensions(self):
        sub_dims = [i.parent for v in self.sub_iterators.values() for i in v]
        return filter_ordered(self.intervals.dimensions + sub_dims)

    @cached_property
    def nonderived_directions(self):
        return {k: v for k, v in self.directions.items() if not k.is_Derived}
