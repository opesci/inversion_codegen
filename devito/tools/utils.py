import ctypes
from collections import OrderedDict
from collections.abc import Iterable
from functools import reduce
from itertools import chain, combinations, groupby, product, zip_longest
from operator import attrgetter, mul
import types

import numpy as np
import sympy
from cgen import dtype_to_ctype as cgen_dtype_to_ctype

__all__ = ['prod', 'as_tuple', 'is_integer', 'generator', 'grouper', 'split', 'roundm',
           'powerset', 'invert', 'flatten', 'single_or', 'filter_ordered', 'as_mapper',
           'filter_sorted', 'dtype_to_cstr', 'dtype_to_ctype', 'dtype_to_mpitype',
           'ctypes_to_cstr', 'ctypes_pointer', 'pprint', 'sweep', 'all_equal', 'as_list',
           'indices_to_slices', 'indices_to_sections']


def prod(iterable, initial=1):
    return reduce(mul, iterable, initial)


def as_list(item, type=None, length=None):
    """
    Force item to a list.
    """
    return list(as_tuple(item, type=type, length=length))


def as_tuple(item, type=None, length=None):
    """
    Force item to a tuple.

    Partly extracted from: https://github.com/OP2/PyOP2/.
    """
    # Empty list if we get passed None
    if item is None:
        t = ()
    elif isinstance(item, (str, sympy.Function)):
        t = (item,)
    else:
        # Convert iterable to list...
        try:
            t = tuple(item)
        # ... or create a list of a single item
        except (TypeError, NotImplementedError):
            t = (item,) * (length or 1)
    if length and not len(t) == length:
        raise ValueError("Tuple needs to be of length %d" % length)
    if type and not all(isinstance(i, type) for i in t):
        raise TypeError("Items need to be of type %s" % type)
    return t


def as_mapper(iterable, key=None, get=None):
    """
    Rearrange an iterable into a dictionary of lists in which keys are
    produced by the function ``key``.
    """
    key = key or (lambda i: i)
    get = get or (lambda i: i)
    mapper = OrderedDict()
    for i in iterable:
        mapper.setdefault(key(i), []).append(get(i))
    return mapper


def is_integer(value):
    """
    A thorough instance comparison for all integer types.
    """
    return isinstance(value, (int, np.integer, sympy.Integer))


def generator():
    """
    Return a function ``f`` that generates integer numbers starting at 0
    with stepping 1.
    """
    def f():
        ret = f.counter
        f.counter += 1
        return ret
    f.counter = 0
    return f


def grouper(iterable, n):
    """Split an interable into groups of size n, plus a reminder"""
    args = [iter(iterable)] * n
    return ([e for e in t if e is not None] for t in zip_longest(*args))


def all_equal(iterable):
    "Returns True if all the elements are equal to each other"
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def split(iterable, f):
    """Split an iterable ``I`` into two iterables ``I1`` and ``I2`` of the
    same type as ``I``. ``I1`` contains all elements ``e`` in ``I`` for
    which ``f(e)`` returns True; ``I2`` is the complement of ``I1``."""
    i1 = type(iterable)(i for i in iterable if f(i))
    i2 = type(iterable)(i for i in iterable if not f(i))
    return i1, i2


def roundm(x, y):
    """Return x rounded up to the closest multiple of y."""
    return x if x % y == 0 else x + y - x % y


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def invert(mapper):
    """Invert a dict of lists preserving the order."""
    inverse = OrderedDict()
    for k, v in mapper.items():
        for i in v:
            inverse[i] = k
    return inverse


def flatten(l):
    """Flatten a hierarchy of nested lists into a plain list."""
    newlist = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes, np.ndarray)):
            for sub in flatten(el):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist


def single_or(l):
    """Return True iff only one item is different than ``None``, False otherwise.
    Note that this is not a XOR function, according to the truth table of the XOR
    boolean function with n > 2 inputs. Hence the name ``single_or``."""
    # No
    i = iter(l)
    return any(i) and not any(i)


def filter_ordered(elements, key=None):
    """
    Filter elements in a list while preserving order.

    Parameters
    ----------
    key : callable, optional
        Conversion key used during equality comparison.
    """
    if isinstance(elements, types.GeneratorType):
        elements = list(elements)
    seen = set()
    if key is None:
        try:
            unordered, inds = np.unique(elements, return_index=True)
            return unordered[np.argsort(inds)].tolist()
        except:
            return sorted(list(set(elements)), key=elements.index)
    else:
        ret = []
        for e in elements:
            k = key(e)
            if k not in seen:
                ret.append(e)
                seen.add(k)
        return ret


def filter_sorted(elements, key=None):
    """
    Filter elements in a list and sort them by key. The default key is
    ``operator.attrgetter('name')``.
    """
    if key is None:
        key = attrgetter('name')
    return sorted(filter_ordered(elements, key=key), key=key)


def dtype_to_cstr(dtype):
    """Translate numpy.dtype into C string."""
    return cgen_dtype_to_ctype(dtype)


def dtype_to_ctype(dtype):
    """Translate numpy.dtype into a ctypes type."""
    return {np.int32: ctypes.c_int,
            np.float32: ctypes.c_float,
            np.int64: ctypes.c_int64,
            np.float64: ctypes.c_double}[dtype]


def dtype_to_mpitype(dtype):
    """Map numpy types to MPI datatypes."""
    return {np.int32: 'MPI_INT',
            np.float32: 'MPI_FLOAT',
            np.int64: 'MPI_LONG',
            np.float64: 'MPI_DOUBLE'}[dtype]


def ctypes_to_cstr(ctype, toarray=None):
    """Translate ctypes types into C strings."""
    if issubclass(ctype, ctypes.Structure):
        return 'struct %s' % ctype.__name__
    elif issubclass(ctype, ctypes.Union):
        return 'union %s' % ctype.__name__
    elif issubclass(ctype, ctypes._Pointer):
        if toarray:
            return ctypes_to_cstr(ctype._type_, '(* %s)' % toarray)
        else:
            return '%s *' % ctypes_to_cstr(ctype._type_)
    elif issubclass(ctype, ctypes.Array):
        return '%s[%d]' % (ctypes_to_cstr(ctype._type_, toarray), ctype._length_)
    elif ctype.__name__.startswith('c_'):
        if ctype.__name__.startswith('c_volatile_'):
            name = 'volatile %s' % ctype.__name__[11:]
        else:
            name = ctype.__name__[2:]
        # A primitive datatype
        # FIXME: Is there a better way of extracting the C typename ?
        # Here, we're following the ctypes convention that each basic type has
        # either the format c_X_p or c_X, where X is the C typename, for instance
        # `int` or `float`.
        if name.endswith('_p'):
            return '%s *' % name[:-2]
        elif toarray:
            return '%s %s' % (name, toarray)
        else:
            return name
    else:
        # A custom datatype (e.g., a typedef-ed pointer to struct)
        return ctype.__name__


def ctypes_pointer(name):
    """Create a ctypes type representing a C pointer to a custom data type ``name``."""
    return type("c_%s_p" % name, (ctypes.c_void_p,), {})


def pprint(node, verbose=True):
    """
    Shortcut to pretty print Iteration/Expression trees.
    """
    from devito.ir.iet import printAST
    print(printAST(node, verbose))


def sweep(parameters, keys=None):
    """
    Generator to create a parameter sweep from a dictionary of values
    or value lists.
    """
    keys = keys or parameters.keys()
    sweep_values = [parameters[key] for key in keys]
    # Ensure all values are iterables to make sweeping safe
    sweep_values = [[v] if isinstance(v, str) or not isinstance(v, Iterable) else v
                    for v in sweep_values]
    for vals in product(*sweep_values):
        yield dict(zip(keys, vals))


def indices_to_slices(inputlist):
    """
    Convert a flatten list of indices to a list of slices.

    Extracted from:
        https://stackoverflow.com/questions/10987777/\
                python-converting-a-list-of-indices-to-slices

    Examples
    --------
    >>> indices_to_slices([0,2,3,4,5,6,12,99,100,101,102,13,14,18,19,20,25])
    [(0, 1), (2, 7), (12, 15), (18, 21), (25, 26), (99, 103)]
    """
    inputlist.sort()
    pointers = np.where(np.diff(inputlist) > 1)[0]
    pointers = zip(np.r_[0, pointers+1], np.r_[pointers, len(inputlist)-1])
    slices = [(inputlist[i], inputlist[j]+1) for i, j in pointers]
    return slices


def indices_to_sections(inputlist):
    """
    Convert a flatten list of indices to a list of sections.

    A section is a (start, size) tuple.

    Examples
    --------
    >>> indices_to_sections([0,2,3,4,5,6,12,99,100,101,102,13,14,18,19,20,25])
    [(0, 1), (2, 5), (12, 3), (18, 3), (25, 1), (99, 4)]
    """
    slices = indices_to_slices(inputlist)
    sections = [(i, j - i) for i, j in slices]
    return sections
