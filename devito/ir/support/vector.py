from collections import OrderedDict

from sympy import true

from devito.symbolics import q_positive
from devito.tools import as_tuple, is_integer, memoized_meth
from devito.types import Dimension

__all__ = ['Vector', 'LabeledVector', 'vmin', 'vmax']


class Vector(tuple):

    """
    An object in an N-dimensional space.

    The elements of a vector can be anything as long as they support the
    comparison operators (`__eq__`, `__lt__`, ...). Also, the `__sub__`
    operator must be available.

    Notes
    -----
    1) Comparison of a Vector with a scalar
    If a comparison between a vector and a non-vector is attempted, then the
    non-vector is promoted to a vector; if this is not possible, an exception
    is raised. This is handy because it turns a vector-scalar comparison into
    a vector-vector comparison with the scalar broadcasted to as many vector
    entries as necessary. For example:

        (3, 4, 5) > 4 => (3, 4, 5) > (4, 4, 4) => False

    2) Comparison of Vectors whose elements are SymPy expressions
    We treat vectors of SymPy expressions as a very special case. When we
    compare two elements, it might not be possible to determine the truth value
    of the relation. For example, the truth value of `3*i < 4*j` cannot be
    determined (unless some information about `i` and `j` is available). In
    some cases, however, the comparison is feasible; for example, `i + 4 < i`
    is definitely False. A sufficient condition for two Vectors to be
    comparable is that their pair-wise indices are affine functions of the same
    variables, with identical coefficient.  If the Vector is instantiated
    passing the keyword argument ``smart = True``, some manipulation will be
    attempted to infer the truth value of a non-trivial symbolic relation. This
    increases the cost of the comparison (and not always an answer may be
    derived), so use it judiciously. By default, ``smart = False``.

    Raises
    ------
    TypeError
        If two Vectors cannot be compared, e.g. due to incomparable symbolic entries.
    """

    def __new__(cls, *items, smart=False):
        obj = super(Vector, cls).__new__(cls, items)
        obj.smart = smart
        return obj

    def _asvector(relax=False):
        def __asvector(func):
            def wrapper(self, other):
                if not isinstance(other, Vector):
                    try:
                        other = Vector(*other)
                    except TypeError:
                        # Not iterable
                        other = Vector(*(as_tuple(other)*len(self)))
                if relax is False and len(self) != len(other):
                    raise TypeError("Cannot operate with Vectors of different rank")
                return func(self, other)
            return wrapper
        return __asvector

    def __hash__(self):
        return super(Vector, self).__hash__()

    @_asvector()
    def __add__(self, other):
        return Vector(*[i + j for i, j in zip(self, other)], smart=self.smart)

    @_asvector()
    def __radd__(self, other):
        return self + other

    @_asvector()
    def __sub__(self, other):
        return Vector(*[i - j for i, j in zip(self, other)], smart=self.smart)

    @_asvector()
    def __rsub__(self, other):
        return self - other

    @_asvector(relax=True)
    def __eq__(self, other):
        return super(Vector, self).__eq__(other)

    @_asvector(relax=True)
    def __ne__(self, other):
        return super(Vector, self).__ne__(other)

    def __lt__(self, other):
        # This might raise an exception if the distance between the i-th entry
        # of `self` and `other` isn't integer, but rather a generic expression
        # not comparable to 0. However, the implementation is "smart", in the
        # sense that it will return as soon as the first two comparable entries
        # (i.e., such that their distance is a non-zero integer) are found
        for i in self.distance(other):
            try:
                val = int(i)
                if val < 0:
                    return True
                elif val > 0:
                    return False
            except TypeError:
                if self.smart:
                    if (i < 0) == true:
                        return True
                    elif (i <= 0) == true:
                        # If `i` can assume the value 0 in at least one case, then
                        # definitely `i < 0` is generally False, so __lt__ must
                        # return False
                        return False
                    elif (i >= 0) == true:
                        return False
                raise TypeError("Non-comparable index functions")

        return False

    def __gt__(self, other):
        # This method is "symmetric" to `__lt__`, but instead of just returning
        # `other.__lt__(self)` we implement it explicitly because this way we
        # can avoid computing the distance in the special case `other is 0`

        # This might raise an exception if the distance between the i-th entry
        # of `self` and `other` isn't integer, but rather a generic expression
        # not comparable to 0. However, the implementation is "smart", in the
        # sense that it will return as soon as the first two comparable entries
        # (i.e., such that their distance is a non-zero integer) are found
        for i in self.distance(other):
            try:
                val = int(i)
                if val > 0:
                    return True
                elif val < 0:
                    return False
            except TypeError:
                if self.smart:
                    if (i > 0) == true:
                        return True
                    elif (i >= 0) == true:
                        # If `i` can assume the value 0 in at least one case, then
                        # definitely `i > 0` is generally False, so __gt__ must
                        # return False
                        return False
                    elif (i <= 0) == true:
                        return False
                    elif q_positive(i):
                        return True
                raise TypeError("Non-comparable index functions")

        return False

    def __le__(self, other):
        if self.__eq__(other):
            return True

        # We cannot simply resort to `__lt__` as it might happen that:
        # * v0 < v1 --> False
        # * v0 == v1 --> False
        # But
        # * v0 <= v1 --> True
        #
        # For example, take `v0 = (a + 2)` and `v1 = (2)`; if `a` is attached
        # the property that definitely `a >= 0`, then surely `v1 <= v0`, even
        # though it can't be assumed anything about `v1 < 0` and `v1 == v0`
        for i in self.distance(other):
            try:
                val = int(i)
                if val < 0:
                    return True
                elif val > 0:
                    return False
            except TypeError:
                if self.smart:
                    if (i < 0) == true:
                        return True
                    elif (i <= 0) == true:
                        continue
                    elif (i > 0) == true:
                        return False
                    elif (i >= 0) == true:
                        # See analogous considerations in __lt__
                        return False
                raise TypeError("Non-comparable index functions")

        # Note: unlike `__lt__`, if we end up here, then *it is* <=. For example,
        # with `v0` and `v1` as above, we would get here
        return True

    @_asvector()
    def __ge__(self, other):
        return other.__le__(self)

    def __getitem__(self, key):
        ret = super(Vector, self).__getitem__(key)
        return Vector(*ret, smart=self.smart) if isinstance(key, slice) else ret

    def __repr__(self):
        return "(%s)" % ','.join(str(i) for i in self)

    @property
    def rank(self):
        return len(self)

    @property
    def sum(self):
        return sum(self)

    @property
    def is_constant(self):
        return all(is_integer(i) for i in self)

    def distance(self, other):
        """
        Compute the distance from ``self`` to ``other``.

        The distance is a reflexive, transitive, and anti-symmetric relation,
        which establishes a total ordering amongst Vectors.

        The distance is a function [Vector x Vector --> D]. D is a tuple of length
        equal to the Vector ``rank``. The i-th entry of D, D_i, indicates whether
        the i-th component of ``self``, self_i, precedes (< 0), equals (== 0), or
        succeeds (> 0) the i-th component of ``other``, other_i.

        In particular, the *absolute value* of D_i represents the number of
        integer points that exist between self_i and sink_i.

        Examples
        --------
                 | 3 |           | 1 |               |  2  |
        source = | 2 | ,  sink = | 4 | , distance => | -2  |
                 | 1 |           | 5 |               | -4  |

        There are 2, 2, and 4 points between [3-2], [2-4], and [1-5], respectively.
        """
        try:
            # Handle quickly the special (yet relevant) cases `other == 0`
            if is_integer(other) and other == 0:
                return self
            elif all(i == 0 for i in other) and self.rank == other.rank:
                return self
        except TypeError:
            pass

        return self - other


class LabeledVector(Vector):

    """
    A Vector that associates a Dimension to each element.
    """

    def __new__(cls, items=None):
        try:
            labels, values = zip(*items)
        except (ValueError, TypeError):
            labels, values = (), ()
        if not all(isinstance(i, Dimension) for i in labels):
            raise ValueError("All labels must be of type Dimension, got [%s]"
                             % ','.join(i.__class__.__name__ for i in labels))
        obj = super(LabeledVector, cls).__new__(cls, *values)
        obj.labels = labels
        return obj

    @classmethod
    def transpose(cls, *vectors):
        """
        Transpose a matrix represented as an iterable of homogeneous LabeledVectors.
        """
        if len(vectors) == 0:
            return LabeledVector()
        if not all(isinstance(v, LabeledVector) for v in vectors):
            raise ValueError("All items must be of type LabeledVector, got [%s]"
                             % ','.join(i.__class__.__name__ for i in vectors))
        T = OrderedDict()
        for v in vectors:
            for l, i in zip(v.labels, v):
                T.setdefault(l, []).append(i)
        return tuple((l, Vector(*i)) for l, i in T.items())

    def __repr__(self):
        return "(%s)" % ','.join('%s:%s' % (l, i) for l, i in zip(self.labels, self))

    def __hash__(self):
        return hash((tuple(self), self.labels))

    def __eq__(self, other):
        if isinstance(other, LabeledVector) and self.labels != other.labels:
            raise TypeError("Cannot compare due to mismatching `labels`")
        return super(LabeledVector, self).__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if isinstance(other, LabeledVector) and self.labels != other.labels:
            raise TypeError("Cannot compare due to mismatching `labels`")
        return super(LabeledVector, self).__lt__(other)

    def __gt__(self, other):
        return other.__lt__(self)

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __getitem__(self, index):
        if isinstance(index, (slice, int)):
            return super(LabeledVector, self).__getitem__(index)
        elif isinstance(index, Dimension):
            for d in self.labels:
                if d._defines & index._defines:
                    i = self.labels.index(d)
                    return super(LabeledVector, self).__getitem__(i)
            return None
        else:
            raise TypeError("Indices must be integers, slices, or Dimensions, not %s"
                            % type(index))

    def fromlabel(self, label, v=None):
        return self[label] if label in self.labels else v

    def items(self):
        return zip(self.labels, self)

    @memoized_meth
    def distance(self, other):
        """
        Compute the distance from ``self`` to ``other``.

        Parameters
        ----------
        other : LabeledVector
            The LabeledVector from which the distance is computed.
        """
        if not isinstance(other, LabeledVector):
            raise TypeError("Cannot compute distance from obj of type %s", type(other))
        if self.labels != other.labels:
            raise TypeError("Cannot compute distance due to mismatching `labels`")
        return LabeledVector(list(zip(self.labels, self - other)))


# Utility functions

def vmin(*vectors):
    """
    Retrieve the minimum out of an iterable of Vectors.

    Raises
    ------
    TypeError
        If there are two incomparable Vectors.
    ValueError
        If an empty sequence is supplied
    """
    if not all(isinstance(i, Vector) for i in vectors):
        raise TypeError("Expected an iterable of Vectors")
    if len(vectors) == 0:
        raise ValueError("min() arg is an empty sequence")
    ret = vectors[0]
    for i in vectors[1:]:
        if i < ret or i <= ret:
            ret = i
    return ret


def vmax(*vectors):
    """
    Retrieve the maximum out of an iterable of Vectors.

    Raises
    ------
    TypeError
        If there are two incomparable Vectors.
    ValueError
        If an empty sequence is supplied
    """
    if not all(isinstance(i, Vector) for i in vectors):
        raise TypeError("Expected an iterable of Vectors")
    if len(vectors) == 0:
        raise ValueError("min() arg is an empty sequence")
    ret = vectors[0]
    for i in vectors[1:]:
        if i > ret or i >= ret:
            ret = i
    return ret
