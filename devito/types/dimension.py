from collections import namedtuple

import sympy
import numpy as np
from cached_property import cached_property

from devito.data import LEFT, RIGHT
from devito.exceptions import InvalidArgument
from devito.logger import debug
from devito.tools import Pickable, dtype_to_cstr, is_integer
from devito.types.args import ArgProvider
from devito.types.basic import Symbol, DataSymbol, Scalar

__all__ = ['Dimension', 'SpaceDimension', 'TimeDimension', 'DefaultDimension',
           'CustomDimension', 'SteppingDimension', 'SubDimension', 'ConditionalDimension',
           'dimensions', 'ModuloDimension', 'IncrDimension']


Thickness = namedtuple('Thickness', 'left right')
SubDimensionOffset = namedtuple('SubDimensionOffset', 'value extreme thickness')


class Dimension(ArgProvider):

    """
    Symbol defining an iteration space.

    A Dimension represents a problem dimension. It is typically used to index
    into Functions, but it can also appear in the middle of a symbolic expression
    just like any other symbol.

    Dimension is the root of a hierarchy of classes, which looks as follows (only
    the classes exposed to the level of the user API are shown)::

                                       Dimension
                                           |
                              ---------------------------
                              |                         |
                       BasicDimension            DefaultDimension
                              |
                      DerivedDimension
                              |
            ---------------------------------------
            |                 |                   |
      SteppingDimension   SubDimension   ConditionalDimension

    Parameters
    ----------
    name : str
        Name of the dimension.
    spacing : symbol, optional
        A symbol to represent the physical spacing along this Dimension.

    Examples
    --------
    Dimensions are automatically created when a Grid is instantiated.

    >>> from devito import Grid
    >>> grid = Grid(shape=(4, 4))
    >>> x, y = grid.dimensions
    >>> type(x)
    <class 'devito.types.dimension.SpaceDimension'>
    >>> time = grid.time_dim
    >>> type(time)
    <class 'devito.types.dimension.TimeDimension'>
    >>> t = grid.stepping_dim
    >>> type(t)
    <class 'devito.types.dimension.SteppingDimension'>

    Alternatively, one can create Dimensions explicitly

    >>> from devito import Dimension
    >>> i = Dimension(name='i')

    Or, when many "free" Dimensions are needed, with the shortcut

    >>> from devito import dimensions
    >>> i, j, k = dimensions('i j k')

    A Dimension can be used to build a Function as well as within symbolic
    expressions, as both array index ("indexed notation") and free symbol.

    >>> from devito import Function
    >>> f = Function(name='f', shape=(4, 4), dimensions=(i, j))
    >>> f + f
    2*f(i, j)
    >>> f[i + 1, j] + f[i, j + 1]
    f[i, j + 1] + f[i + 1, j]
    >>> f*i
    i*f(i, j)
    """

    is_Dimension = True
    is_Space = False
    is_Time = False

    is_Default = False
    is_Custom = False
    is_Derived = False
    is_NonlinearDerived = False
    is_Sub = False
    is_Conditional = False
    is_Stepping = False
    is_Modulo = False
    is_Incr = False

    _C_typename = 'const %s' % dtype_to_cstr(np.int32)
    _C_typedata = _C_typename

    def __new__(cls, *args, **kwargs):
        """
        Equivalent to ``BasicDimension(*args, **kwargs)``.

        Notes
        -----
        This is only necessary for backwards compatibility, as originally
        there was no BasicDimension (i.e., Dimension was just the top class).
        """
        if cls is Dimension:
            return BasicDimension(*args, **kwargs)
        else:
            return BasicDimension.__new__(cls, *args, **kwargs)

    @classmethod
    def class_key(cls):
        """
        Overrides sympy.Symbol.class_key such that Dimensions always
        preceed other symbols when printed (e.g. x + h_x, not h_x + x).
        """
        a, b, c = super().class_key()
        return a, b - 1, c

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        # Unlike other Symbols, Dimensions can only be integers
        return np.int32

    def __str__(self):
        return self.name

    @property
    def spacing(self):
        """Symbol representing the physical spacing along the Dimension."""
        return self._spacing

    @cached_property
    def symbolic_size(self):
        """Symbolic size of the Dimension."""
        return Scalar(name=self.size_name, dtype=np.int32, is_const=True)

    @cached_property
    def symbolic_min(self):
        """Symbol defining the minimum point of the Dimension."""
        return Scalar(name=self.min_name, dtype=np.int32, is_const=True)

    @cached_property
    def symbolic_max(self):
        """Symbol defining the maximum point of the Dimension."""
        return Scalar(name=self.max_name, dtype=np.int32, is_const=True)

    @property
    def symbolic_incr(self):
        """The increment value while iterating over the Dimension."""
        return sympy.S.One

    @cached_property
    def size_name(self):
        return "%s_size" % self.name

    @cached_property
    def min_name(self):
        return "%s_m" % self.name

    @cached_property
    def max_name(self):
        return "%s_M" % self.name

    @property
    def is_const(self):
        return False

    @property
    def root(self):
        return self

    @property
    def _maybe_distributed(self):
        """Could it be a distributed Dimension?"""
        return True

    @property
    def _C_name(self):
        return self.name

    @cached_property
    def _defines(self):
        return frozenset({self})

    @cached_property
    def _defines_symbols(self):
        candidates = [self.symbolic_min, self.symbolic_max, self.symbolic_size,
                      self.symbolic_incr]
        return frozenset(i for i in candidates if not i.is_Number)

    @property
    def _arg_names(self):
        """Tuple of argument names introduced by the Dimension."""
        return (self.name, self.size_name, self.max_name, self.min_name)

    def _arg_defaults(self, _min=None, size=None, alias=None):
        """
        A map of default argument values defined by the Dimension.

        Parameters
        ----------
        _min : int, optional
            Minimum point as provided by data-carrying objects.
        size : int, optional
            Size as provided by data-carrying symbols.
        alias : Dimension, optional
            To get the min/max/size names under which to store values. Use
            self's if None.
        """
        dim = alias or self
        return {dim.min_name: _min or 0,
                dim.size_name: size,
                dim.max_name: size if size is None else size-1}

    def _arg_values(self, args, interval, grid, **kwargs):
        """
        Produce a map of argument values after evaluating user input. If no user
        input is provided, get a known value in ``args`` and adjust it so that no
        out-of-bounds memory accesses will be performeed. The adjustment exploits
        the information in ``interval``, an Interval describing the Dimension data
        space. If no value is available in ``args``, use a default value.

        Parameters
        ----------
        args : dict
            Known argument values.
        interval : Interval
            Description of the Dimension data space.
        grid : Grid
            Used for spacing overriding and MPI execution; if ``self`` is a distributed
            Dimension, then ``grid`` is used to translate user input into rank-local
            indices.
        **kwargs
            Dictionary of user-provided argument overrides.
        """
        # Fetch user input and convert into rank-local values
        glb_minv = kwargs.pop(self.min_name, None)
        glb_maxv = kwargs.pop(self.max_name, kwargs.pop(self.name, None))
        if grid is not None and grid.is_distributed(self):
            loc_minv, loc_maxv = grid.distributor.glb_to_loc(self, (glb_minv, glb_maxv))
        else:
            loc_minv, loc_maxv = glb_minv, glb_maxv

        # If no user-override provided, use a suitable default value
        defaults = self._arg_defaults()
        if glb_minv is None:
            loc_minv = args.get(self.min_name, defaults[self.min_name])
            try:
                loc_minv -= min(interval.lower, 0)
            except (AttributeError, TypeError):
                pass
        if glb_maxv is None:
            loc_maxv = args.get(self.max_name, defaults[self.max_name])
            try:
                loc_maxv -= max(interval.upper, 0)
            except (AttributeError, TypeError):
                pass

        return {self.min_name: loc_minv, self.max_name: loc_maxv}

    def _arg_check(self, args, size, interval):
        """
        Raises
        ------
        InvalidArgument
            If any of the ``self``-related runtime arguments in ``args``
            will cause an out-of-bounds access.
        """
        if self.min_name not in args:
            raise InvalidArgument("No runtime value for %s" % self.min_name)
        if interval.is_Defined and args[self.min_name] + interval.lower < 0:
            raise InvalidArgument("OOB detected due to %s=%d" % (self.min_name,
                                                                 args[self.min_name]))

        if self.max_name not in args:
            raise InvalidArgument("No runtime value for %s" % self.max_name)
        if interval.is_Defined:
            if is_integer(interval.upper):
                upper = interval.upper
            else:
                # Autopadding causes non-integer upper limit
                upper = interval.upper.subs(args)
            if args[self.max_name] + upper >= size:
                raise InvalidArgument("OOB detected due to %s=%d" % (self.max_name,
                                                                     args[self.max_name]))

        # Allow the specific case of max=min-1, which disables the loop
        if args[self.max_name] < args[self.min_name]-1:
            raise InvalidArgument("Illegal %s=%d < %s=%d"
                                  % (self.max_name, args[self.max_name],
                                     self.min_name, args[self.min_name]))
        elif args[self.max_name] == args[self.min_name]-1:
            debug("%s=%d and %s=%d might cause no iterations along Dimension %s",
                  self.min_name, args[self.min_name],
                  self.max_name, args[self.max_name], self.name)

    # Pickling support
    _pickle_args = ['name']
    _pickle_kwargs = ['spacing']
    __reduce_ex__ = Pickable.__reduce_ex__


class BasicDimension(Dimension, Symbol):

    __doc__ = Dimension.__doc__

    def __new__(cls, *args, **kwargs):
        return Symbol.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, name, spacing=None):
        self._spacing = spacing or Scalar(name='h_%s' % name, is_const=True)


class DefaultDimension(Dimension, DataSymbol):

    """
    Symbol defining an iteration space with statically-known size.

    Parameters
    ----------
    name : str
        Name of the dimension.
    spacing : Symbol, optional
        A symbol to represent the physical spacing along this Dimension.
    default_value : float, optional
        Default value associated with the Dimension.

    Notes
    -----
    A DefaultDimension carries a value, so it has a mutable state. Hence, it is
    not cached.
    """

    is_Default = True

    def __new__(cls, *args, **kwargs):
        return DataSymbol.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, name, spacing=None, default_value=None):
        self._spacing = spacing or Scalar(name='h_%s' % name, is_const=True)
        self._default_value = default_value or 0

    @cached_property
    def symbolic_size(self):
        return sympy.Number(self._default_value)

    def _arg_defaults(self, _min=None, size=None, alias=None):
        dim = alias or self
        size = size or dim._default_value
        return {dim.min_name: _min or 0, dim.size_name: size,
                dim.max_name: size if size is None else size-1}


class SpaceDimension(BasicDimension):

    """
    Symbol defining an iteration space.

    This symbol represents a space dimension that defines the extent of
    a physical grid.

    A SpaceDimension creates dedicated shortcut notations for spatial
    derivatives on Functions.

    Parameters
    ----------
    name : str
        Name of the dimension.
    spacing : symbol, optional
        A symbol to represent the physical spacing along this Dimension.
    """

    is_Space = True


class TimeDimension(BasicDimension):

    """
    Symbol defining an iteration space.

    This symbol represents a time dimension that defines the extent of time.

    A TimeDimension create dedicated shortcut notations for time derivatives
    on Functions.

    Parameters
    ----------
    name : str
        Name of the dimension.
    spacing : symbol, optional
        A symbol to represent the physical spacing along this Dimension.
    """

    is_Time = True


class DerivedDimension(BasicDimension):

    """
    Symbol defining an iteration space derived from a ``parent`` Dimension.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The parent Dimension.
    """

    is_Derived = True

    def __init_finalize__(self, name, parent):
        assert isinstance(parent, Dimension)
        self._parent = parent
        # Inherit time/space identifiers
        self.is_Time = parent.is_Time
        self.is_Space = parent.is_Space

    @property
    def parent(self):
        return self._parent

    @property
    def root(self):
        return self._parent.root

    @property
    def spacing(self):
        return self.parent.spacing

    @cached_property
    def _defines(self):
        return frozenset({self}) | self.parent._defines

    @property
    def _arg_names(self):
        return self.parent._arg_names

    def _arg_check(self, *args):
        """A DerivedDimension performs no runtime checks."""
        return

    # Pickling support
    _pickle_args = Dimension._pickle_args + ['parent']
    _pickle_kwargs = []


class SubDimension(DerivedDimension):

    """
    Symbol defining a convex iteration sub-space derived from a ``parent``
    Dimension.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The parent Dimension.
    left : expr-like
        Symbolic expression providing the left (lower) bound of the
        SubDimension.
    right : expr-like
        Symbolic expression providing the right (upper) bound of the
        SubDimension.
    thickness : 2-tuple of 2-tuples
        The thickness of the left and right regions, respectively.
    local : bool
        True if, in case of domain decomposition, the SubDimension is
        guaranteed not to span more than one domains, False otherwise.

    Examples
    --------
    SubDimensions should *not* be created directly in user code; SubDomains
    should be used instead. Exceptions are rare.

    To create a SubDimension, one should use the shortcut methods ``left``,
    ``right``, ``middle``. For example, to create a SubDimension that spans
    the entire space of the parent Dimension except for the two extremes:

    >>> from devito import Dimension, SubDimension
    >>> x = Dimension('x')
    >>> xi = SubDimension.middle('xi', x, 1, 1)

    For a SubDimension that only spans the three leftmost points of its
    parent Dimension, instead:

    >>> xl = SubDimension.left('xl', x, 3)

    SubDimensions created via the ``left`` and ``right`` shortcuts are, by default,
    local (i.e., non-distributed) Dimensions, as they are assumed to fit entirely
    within a single domain. This is the most typical use case (e.g., to set up
    boundary conditions). To drop this assumption, pass ``local=False``.
    """

    is_Sub = True

    def __init_finalize__(self, name, parent, left, right, thickness, local):
        super().__init_finalize__(name, parent)
        self._interval = sympy.Interval(left, right)
        self._thickness = Thickness(*thickness)
        self._local = local

    @classmethod
    def _symbolic_thickness(cls, name):
        return (Scalar(name="%s_ltkn" % name, dtype=np.int32,
                       is_const=True, nonnegative=True),
                Scalar(name="%s_rtkn" % name, dtype=np.int32,
                       is_const=True, nonnegative=True))

    @classmethod
    def left(cls, name, parent, thickness, local=True):
        lst, rst = cls._symbolic_thickness(name)
        return cls(name, parent,
                   left=parent.symbolic_min,
                   right=parent.symbolic_min+lst-1,
                   thickness=((lst, thickness), (rst, 0)),
                   local=local)

    @classmethod
    def right(cls, name, parent, thickness, local=True):
        lst, rst = cls._symbolic_thickness(name)
        return cls(name, parent,
                   left=parent.symbolic_max-rst+1,
                   right=parent.symbolic_max,
                   thickness=((lst, 0), (rst, thickness)),
                   local=local)

    @classmethod
    def middle(cls, name, parent, thickness_left, thickness_right, local=False):
        lst, rst = cls._symbolic_thickness(name)
        return cls(name, parent,
                   left=parent.symbolic_min+lst,
                   right=parent.symbolic_max-rst,
                   thickness=((lst, thickness_left), (rst, thickness_right)),
                   local=local)

    @cached_property
    def symbolic_min(self):
        return self._interval.left

    @cached_property
    def symbolic_max(self):
        return self._interval.right

    @cached_property
    def symbolic_size(self):
        # The size must be given as a function of the parent's symbols
        return self.symbolic_max - self.symbolic_min + 1

    @property
    def local(self):
        return self._local

    @property
    def thickness(self):
        return self._thickness

    @property
    def _maybe_distributed(self):
        return not self.local

    @cached_property
    def _thickness_map(self):
        return dict(self.thickness)

    @cached_property
    def _defines_symbols(self):
        return super()._defines_symbols | frozenset(self._thickness_map)

    @cached_property
    def _offset_left(self):
        # The left extreme of the SubDimension can be related to either the
        # min or max of the parent dimension
        try:
            symbolic_thickness = self.symbolic_min - self.parent.symbolic_min
            val = symbolic_thickness.subs(self._thickness_map)
            return SubDimensionOffset(
                int(val),
                self.parent.symbolic_min,
                symbolic_thickness
            )
        except TypeError:
            symbolic_thickness = self.symbolic_min - self.parent.symbolic_max
            val = symbolic_thickness.subs(self._thickness_map)
            return SubDimensionOffset(
                int(val),
                self.parent.symbolic_max,
                symbolic_thickness
            )

    @cached_property
    def _offset_right(self):
        # The right extreme of the SubDimension can be related to either the
        # min or max of the parent dimension
        try:
            symbolic_thickness = self.symbolic_max - self.parent.symbolic_min
            val = symbolic_thickness.subs(self._thickness_map)
            return SubDimensionOffset(
                int(val),
                self.parent.symbolic_min,
                symbolic_thickness
            )
        except TypeError:
            symbolic_thickness = self.symbolic_max - self.parent.symbolic_max
            val = symbolic_thickness.subs(self._thickness_map)
            return SubDimensionOffset(
                int(val),
                self.parent.symbolic_max,
                symbolic_thickness
            )

    def overlap(self, other):
        return (isinstance(other, SubDimension) and
                self.root is other.root and
                self._offset_left.extreme is other._offset_left.extreme and
                self._offset_right.extreme is other._offset_right.extreme)

    @property
    def _arg_names(self):
        return tuple(k.name for k, _ in self.thickness) + self.parent._arg_names

    def _arg_defaults(self, grid=None, **kwargs):
        return {}

    def _arg_values(self, args, interval, grid, **kwargs):
        # Allow override of thickness values to disable BCs
        # However, arguments from the user are considered global
        # So overriding the thickness to a nonzero value should not cause
        # boundaries to exist between ranks where they did not before
        requested_ltkn, requested_rtkn = (
            kwargs.get(k.name, v) for k, v in self.thickness
        )

        if grid is not None and grid.is_distributed(self.root):
            # Get local thickness
            if self.local:
                # dimension is of type ``left``/right`` - compute the 'offset'
                # and then add 1 to get the appropriate thickness
                ltkn = grid.distributor.glb_to_loc(self.root, requested_ltkn-1, LEFT)
                rtkn = grid.distributor.glb_to_loc(self.root, requested_rtkn-1, RIGHT)
                ltkn = ltkn+1 if ltkn is not None else 0
                rtkn = rtkn+1 if rtkn is not None else 0
            else:
                # dimension is of type ``middle``
                ltkn = grid.distributor.glb_to_loc(self.root, requested_ltkn,
                                                   LEFT) or 0
                rtkn = grid.distributor.glb_to_loc(self.root, requested_rtkn,
                                                   RIGHT) or 0
        else:
            ltkn = requested_ltkn
            rtkn = requested_rtkn

        return {i.name: v for i, v in zip(self._thickness_map, (ltkn, rtkn))}

    # Pickling support
    _pickle_args = DerivedDimension._pickle_args +\
        ['symbolic_min', 'symbolic_max', 'thickness', 'local']
    _pickle_kwargs = []


class ConditionalDimension(DerivedDimension):

    """
    Symbol defining a non-convex iteration sub-space derived from a ``parent``
    Dimension, implemented by the compiler generating conditional "if-then" code
    within the parent Dimension's iteration space.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The parent Dimension.
    factor : int, optional
        The number of iterations between two executions of the if-branch. If None
        (default), ``condition`` must be provided.
    condition : expr-like, optional
        An arbitrary SymPy expression, typically involving the ``parent``
        Dimension. When it evaluates to True, the if-branch is executed. If None
        (default), ``factor`` must be provided.
    indirect : bool, optional
        If True, use ``condition``, rather than the parent Dimension, to
        index into arrays. A typical use case is when arrays are accessed
        indirectly via the ``condition`` expression. Defaults to False.

    Examples
    --------
    Among the other things, ConditionalDimensions are indicated to implement
    Function subsampling. In the following example, an Operator evaluates the
    Function ``g`` and saves its content into ``f`` every ``factor=4`` iterations.

    >>> from devito import Dimension, ConditionalDimension, Function, Eq, Operator
    >>> size, factor = 16, 4
    >>> i = Dimension(name='i')
    >>> ci = ConditionalDimension(name='ci', parent=i, factor=factor)
    >>> g = Function(name='g', shape=(size,), dimensions=(i,))
    >>> f = Function(name='f', shape=(int(size/factor),), dimensions=(ci,))
    >>> op = Operator([Eq(g, 1), Eq(f, g)])

    The Operator generates the following for-loop (pseudocode)

    .. code-block:: C

        for (int i = i_m; i <= i_M; i += 1) {
          g[i] = 1;
          if (i%4 == 0) {
            f[i / 4] = g[i];
          }
        }

    Another typical use case is when one needs to constrain the execution of
    loop iterations to make sure certain conditions are honoured. The following
    artificial example employs indirect array accesses and uses ConditionalDimension
    to guard against out-of-bounds accesses.

    >>> from sympy import And
    >>> ci = ConditionalDimension(name='ci', parent=i,
    ...                           condition=And(g[i] > 0, g[i] < 4, evaluate=False))
    >>> f = Function(name='f', shape=(int(size/factor),), dimensions=(ci,))
    >>> op = Operator(Eq(f[g[i]], f[g[i]] + 1))

    The Operator generates the following for-loop (pseudocode)

    .. code-block:: C

        for (int i = i_m; i <= i_M; i += 1) {
          if (g[i] > 0 && g[i] < 4) {
            f[g[i]] = f[g[i]] + 1;
          }
        }

    """

    is_NonlinearDerived = True
    is_Conditional = True

    def __init_finalize__(self, name, parent, factor=None, condition=None,
                          indirect=False):
        super().__init_finalize__(name, parent)
        self._factor = factor
        self._condition = condition
        self._indirect = indirect

    @property
    def spacing(self):
        return self.factor * self.parent.spacing

    @property
    def factor(self):
        return self._factor if self._factor is not None else 1

    @property
    def condition(self):
        return self._condition

    @property
    def indirect(self):
        return self._indirect

    @property
    def index(self):
        return self if self.indirect is True else self.parent

    @cached_property
    def free_symbols(self):
        retval = set(super(ConditionalDimension, self).free_symbols)
        if self.condition is not None:
            retval |= self.condition.free_symbols
        return retval

    # Pickling support
    _pickle_kwargs = DerivedDimension._pickle_kwargs + ['factor', 'condition', 'indirect']


class SteppingDimension(DerivedDimension):

    """
    Symbol defining a convex iteration sub-space derived from a ``parent``
    Dimension, which cyclically produces a finite range of values, such
    as ``0, 1, 2, 0, 1, 2, 0, ...`` (also referred to as "modulo buffered
    iteration").

    SteppingDimension is most commonly used to represent a time-stepping Dimension.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The parent Dimension.
    """

    is_NonlinearDerived = True
    is_Stepping = True

    @property
    def symbolic_min(self):
        return self.parent.symbolic_min

    @property
    def symbolic_max(self):
        return self.parent.symbolic_max

    @property
    def _arg_names(self):
        return (self.min_name, self.max_name, self.name) + self.parent._arg_names

    def _arg_defaults(self, _min=None, **kwargs):
        """
        A map of default argument values defined by this dimension.

        Parameters
        ----------
        _min : int, optional
            Minimum point as provided by data-carrying objects.

        Notes
        -----
        A SteppingDimension does not know its max point and therefore
        does not have a size argument.
        """
        return {self.parent.min_name: _min}

    def _arg_values(self, *args, **kwargs):
        """
        The argument values provided by a SteppingDimension are those
        of its parent, as it acts as an alias.
        """
        values = {}

        if self.min_name in kwargs:
            values[self.parent.min_name] = kwargs.pop(self.min_name)

        if self.max_name in kwargs:
            values[self.parent.max_name] = kwargs.pop(self.max_name)

        # Let the dimension name be an alias for `dim_e`
        if self.name in kwargs:
            values[self.parent.max_name] = kwargs.pop(self.name)

        return values


# The Dimensions below are for internal use only


class ModuloDimension(DerivedDimension):

    """
    Dimension symbol representing a non-contiguous sub-region of a given
    ``parent`` Dimension, which cyclically produces a finite range of values,
    such as ``0, 1, 2, 0, 1, 2, 0, ...``.

    When ``modulo=None``, the ModuloDimension degenerates and keeps generating
    the same number such as ``2, 2, 2, 2, 2, ...`` (the actual value depends
    on ``incr``).

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The Dimension from which the ModuloDimension is derived.
    offset : expr-like, optional
        Min value offset. Defaults to 0.
    modulo : int, optional
        The divisor value.
    incr : expr-like, optional
        The iterator increment value. Defaults to ``offset % modulo``.
    origin : expr-like, optional
        The expression -- typically a function of the parent Dimension -- the
        ModuloDimension represents.

    Notes
    -----
    This type should not be instantiated directly in user code; if in need for
    modulo buffered iteration, use SteppingDimension instead.

    About `origin` -- the ModuloDimensions `t0, t1, t2, ...` are generated by the
    compiler to implement modulo iteration along a TimeDimension. For example,
    `t0`'s `origin` may be `t + 0` (where `t` is a SteppingDimension), `t1`'s
    `origin` will then be `t + 1` and so on.
    """

    is_NonlinearDerived = True
    is_Modulo = True

    def __init_finalize__(self, name, parent,
                          offset=None, modulo=None, incr=None, origin=None):
        super().__init_finalize__(name, parent)

        # Sanity check
        assert modulo is not None or incr is not None

        self._offset = offset or 0
        self._modulo = modulo
        self._incr = incr
        self._origin = origin

    @property
    def offset(self):
        return self._offset

    @property
    def modulo(self):
        return self._modulo

    @property
    def incr(self):
        return self._incr

    @property
    def origin(self):
        return self._origin

    @cached_property
    def symbolic_size(self):
        try:
            return sympy.Number(self.modulo)
        except (TypeError, ValueError):
            pass
        try:
            return sympy.Number(self.incr)
        except (TypeError, ValueError):
            return self.incr

    @cached_property
    def symbolic_min(self):
        if self.modulo is not None:
            return self.offset % self.modulo
        # Make sure we return a symbolic object as this point `offset` may well
        # be a pure Python number
        try:
            return sympy.Number(self.offset)
        except (TypeError, ValueError):
            return self.offset

    @cached_property
    def symbolic_incr(self):
        if self._incr is not None:
            incr = self._incr
        else:
            incr = self.offset
        if self.modulo is not None:
            incr = incr % self.modulo
        # Make sure we return a symbolic object as this point `incr` may well
        # be a pure Python number
        try:
            return sympy.Number(incr)
        except (TypeError, ValueError):
            return incr

    def _arg_defaults(self, **kwargs):
        """
        A ModuloDimension provides no arguments, so this method returns an empty dict.
        """
        return {}

    def _arg_values(self, *args, **kwargs):
        """
        A ModuloDimension provides no arguments, so there are no argument values
        to be derived.
        """
        return {}

    # Override SymPy arithmetic operators to exploit properties of modular arithmetic

    def __add__(self, other):
        # Exploit compatibility with addition:
        # `a1 ≡ b1 (mod n) and a2 ≡ b2 (mod n)` => `a1 + a2 ≡ b1 + b2 (mod n)`
        try:
            if self.modulo == other.modulo:
                return self.origin + other.origin
        except (AttributeError, TypeError):
            pass
        return super().__add__(other)

    def __sub__(self, other):
        # Exploit compatibility with subtraction:
        # `a1 ≡ b1 (mod n) and a2 ≡ b2 (mod n)` => `a1 – a2 ≡ b1 – b2 (mod n)`
        try:
            if self.modulo == other.modulo:
                return self.origin - other.origin
        except (AttributeError, TypeError):
            pass
        return super().__sub__(other)

    # Pickling support
    _pickle_kwargs = ['offset', 'modulo', 'incr', 'origin']


class IncrDimension(DerivedDimension):

    """
    Dimension symbol representing a non-contiguous sub-region of a given
    ``parent`` Dimension, with one point every ``step`` points. Thus, if
    ``step == k``, the dimension represents the sequence ``min, min + k,
    min + 2*k, ...``.

    Parameters
    ----------
    name : str
        Name of the dimension.
    parent : Dimension
        The Dimension from which the IncrDimension is derived.
    _min : expr-like
        The minimum point of the Dimension.
    _max : expr-like
        The maximum point of the Dimension.
    step : expr-like, optional
        The distance between two consecutive points. Defaults to the
        symbolic size.
    size : expr-like, optional
        The symbolic size of the Dimension. Defaults to `_max-_min+1`.

    Notes
    -----
    This type should not be instantiated directly in user code.
    """

    is_Incr = True
    is_PerfKnob = True

    def __init_finalize__(self, name, parent, _min, _max, step=None, size=None):
        super().__init_finalize__(name, parent)
        self._min = _min
        self._max = _max
        self._step = step
        self._size = size

    @property
    def size(self):
        return self._size

    @cached_property
    def step(self):
        if self._step is not None:
            return self._step
        else:
            return Scalar(name=self.size_name, dtype=np.int32, is_const=True)

    @cached_property
    def symbolic_size(self):
        if self.size is not None:
            # Make sure we return a symbolic object as the provided size might
            # be for example a pure int
            try:
                return sympy.Number(self.size)
            except (TypeError, ValueError):
                return self._size
        else:
            # The size must be given as a function of the parent's symbols
            return self.symbolic_max - self.symbolic_min + 1

    @cached_property
    def symbolic_min(self):
        # Make sure we return a symbolic object as the provided min might
        # be for example a pure int
        try:
            return sympy.Number(self._min)
        except (TypeError, ValueError):
            return self._min

    @cached_property
    def symbolic_max(self):
        # Make sure we return a symbolic object as the provided max might
        # be for example a pure int
        try:
            return sympy.Number(self._max)
        except (TypeError, ValueError):
            return self._max

    @cached_property
    def symbolic_incr(self):
        try:
            return sympy.Number(self.step)
        except (TypeError, ValueError):
            return self.step

    @cached_property
    def _arg_names(self):
        try:
            return (self.step.name,)
        except AttributeError:
            # `step` not a Symbol
            return ()

    def _arg_defaults(self, **kwargs):
        # TODO: need a heuristic to pick a default incr size
        # TODO: move default value to __new__
        try:
            return {self.step.name: 8}
        except AttributeError:
            # `step` not a Symbol
            return {}

    def _arg_values(self, args, interval, grid, **kwargs):
        try:
            name = self.step.name
        except AttributeError:
            # `step` not a Symbol
            return {}
        if name in kwargs:
            return {name: kwargs.pop(name)}
        elif isinstance(self.parent, IncrDimension):
            # `self` is an IncrDimension within an outer IncrDimension, but
            # no value supplied -> the sub-block will span the entire block
            return {name: args[self.parent.step.name]}
        else:
            value = self._arg_defaults()[name]
            if value <= args[self.root.max_name] - args[self.root.min_name] + 1:
                return {name: value}
            else:
                # Avoid OOB (will end up here only in case of tiny iteration spaces)
                return {name: 1}

    def _arg_check(self, args, *_args):
        try:
            name = self.step.name
        except AttributeError:
            # `step` not a Symbol
            return

        value = args[name]
        if isinstance(self.parent, IncrDimension):
            # sub-IncrDimensions must be perfect divisors of their parent
            parent_value = args[self.parent.step.name]
            if parent_value % value > 0:
                raise InvalidArgument("Illegal block size `%s=%d`: sub-block sizes "
                                      "must divide the parent block size evenly (`%s=%d`)"
                                      % (name, value, self.parent.step.name,
                                         parent_value))
        else:
            if value < 0:
                raise InvalidArgument("Illegal block size `%s=%d`: it should be > 0"
                                      % (name, value))
            if value > args[self.root.max_name] - args[self.root.min_name] + 1:
                # Avoid OOB
                raise InvalidArgument("Illegal block size `%s=%d`: it's greater than the "
                                      "iteration range and it will cause an OOB access"
                                      % (name, value))

    # Pickling support
    _pickle_args = ['name', 'parent', 'symbolic_min', 'symbolic_max']
    _pickle_kwargs = ['step', 'size']


class CustomDimension(BasicDimension):

    """
    Dimension defining an iteration space with custom (known) size.

    Notes
    -----
    This could be extended in the future for more customization (e.g., incr.).
    """

    is_Custom = True

    def __init_finalize__(self, name, symbolic_min=None, symbolic_max=None,
                          symbolic_size=None, parent=None):
        self._symbolic_min = symbolic_min
        self._symbolic_max = symbolic_max
        self._symbolic_size = symbolic_size
        self._parent = parent
        super().__init_finalize__(name)

    @property
    def is_Derived(self):
        return self._parent is not None

    @property
    def parent(self):
        return self._parent

    @property
    def root(self):
        if self.is_Derived:
            return self.parent.root
        else:
            return self

    @property
    def spacing(self):
        if self.is_Derived:
            return self.parent.spacing
        else:
            return self._spacing

    @cached_property
    def _defines(self):
        ret = frozenset({self})
        if self.is_Derived:
            ret |= self.parent._defines
        return ret

    @property
    def symbolic_min(self):
        try:
            return sympy.Number(self._symbolic_min)
        except (TypeError, ValueError):
            pass
        if self._symbolic_min is None:
            return super().symbolic_min
        else:
            return self._symbolic_min

    @property
    def symbolic_max(self):
        try:
            return sympy.Number(self._symbolic_max)
        except (TypeError, ValueError):
            pass
        if self._symbolic_max is None:
            return super().symbolic_max
        else:
            return self._symbolic_max

    @property
    def symbolic_size(self):
        try:
            return sympy.Number(self._symbolic_size)
        except (TypeError, ValueError):
            pass
        if self._symbolic_size is None:
            return super().symbolic_size
        else:
            return self._symbolic_size

    def _arg_defaults(self, **kwargs):
        return {}

    def _arg_values(self, *args, **kwargs):
        return {}

    def _arg_check(self, *args):
        """A CustomDimension performs no runtime checks."""
        return

    # Pickling support
    _pickle_kwargs = ['symbolic_min', 'symbolic_max', 'symbolic_size', 'parent']


def dimensions(names):
    assert type(names) == str
    return tuple(Dimension(i) for i in names.split())
