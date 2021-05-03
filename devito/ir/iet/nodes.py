"""The Iteration/Expression Tree (IET) hierarchy."""

import abc
import inspect
import numbers
from cached_property import cached_property
from collections import OrderedDict, namedtuple
from collections.abc import Iterable

import cgen as c

from devito.data import FULL
from devito.ir.equations import DummyEq
from devito.ir.support import (SEQUENTIAL, PARALLEL, PARALLEL_IF_ATOMIC,
                               PARALLEL_IF_PVT, VECTORIZED, AFFINE, COLLAPSED,
                               Property, Forward, detect_io)
from devito.symbolics import ListInitializer, FunctionFromPointer, as_symbol, ccode
from devito.tools import Signer, as_tuple, filter_ordered, filter_sorted, flatten
from devito.types.basic import AbstractFunction, Indexed, LocalObject, Symbol

__all__ = ['Node', 'Block', 'Expression', 'Element', 'Callable', 'Call', 'Conditional',
           'Iteration', 'List', 'LocalExpression', 'Section', 'TimedList', 'Prodder',
           'MetaCall', 'PointerCast', 'ForeignExpression', 'HaloSpot', 'IterationTree',
           'ExpressionBundle', 'AugmentedExpression', 'Increment', 'Return', 'While',
           'ParallelIteration', 'ParallelBlock', 'Dereference', 'Lambda', 'SyncSpot',
           'PragmaList', 'DummyExpr', 'BlankLine', 'ParallelTree', 'Break']

# First-class IET nodes


class Node(Signer):

    __metaclass__ = abc.ABCMeta

    is_Node = True
    is_Block = False
    is_Iteration = False
    is_IterationFold = False
    is_While = False
    is_Expression = False
    is_Increment = False
    is_ForeignExpression = False
    is_LocalExpression = False
    is_Callable = False
    is_Lambda = False
    is_ElementalFunction = False
    is_Call = False
    is_List = False
    is_PointerCast = False
    is_Dereference = False
    is_Element = False
    is_Section = False
    is_HaloSpot = False
    is_ExpressionBundle = False
    is_Break = False
    is_ParallelIteration = False
    is_ParallelBlock = False
    is_SyncSpot = False

    _traversable = []
    """
    :attr:`_traversable`. The traversable fields of the Node; that is, fields
    walked over by a Visitor. All arguments in __init__ whose name
    appears in this list are treated as traversable fields.
    """

    def __new__(cls, *args, **kwargs):
        obj = super(Node, cls).__new__(cls)
        argnames, _, _, defaultvalues, _, _, _ = inspect.getfullargspec(cls.__init__)
        try:
            defaults = dict(zip(argnames[-len(defaultvalues):], defaultvalues))
        except TypeError:
            # No default kwarg values
            defaults = {}
        obj._args = {k: v for k, v in zip(argnames[1:], args)}
        obj._args.update(kwargs.items())
        obj._args.update({k: defaults.get(k) for k in argnames[1:] if k not in obj._args})
        return obj

    def _rebuild(self, *args, **kwargs):
        """Reconstruct ``self``."""
        handle = self._args.copy()  # Original constructor arguments
        argnames = [i for i in self._traversable if i not in kwargs]
        handle.update(OrderedDict([(k, v) for k, v in zip(argnames, args)]))
        handle.update(kwargs)
        return type(self)(**handle)

    @cached_property
    def ccode(self):
        """
        Generate C code.

        This is a shorthand for

            .. code-block:: python

              from devito.ir.iet import CGen
              CGen().visit(self)
        """
        from devito.ir.iet.visitors import CGen
        return CGen().visit(self)

    @property
    def view(self):
        """A representation of the IET rooted in ``self``."""
        from devito.ir.iet.visitors import printAST
        return printAST(self)

    @property
    def children(self):
        """Return the traversable children."""
        return tuple(getattr(self, i) for i in self._traversable)

    @property
    def args(self):
        """Arguments used to construct the Node."""
        return self._args.copy()

    @property
    def args_frozen(self):
        """Arguments used to construct the Node that cannot be traversed."""
        return {k: v for k, v in self.args.items() if k not in self._traversable}

    def __str__(self):
        return str(self.ccode)

    @abc.abstractproperty
    def functions(self):
        """All AbstractFunction objects used by this node."""
        raise NotImplementedError()

    @abc.abstractproperty
    def free_symbols(self):
        """All Symbol objects used by this node."""
        raise NotImplementedError()

    @abc.abstractproperty
    def defines(self):
        """All Symbol objects defined by this node."""
        raise NotImplementedError()

    def _signature_items(self):
        return (str(self.ccode),)


class ExprStmt(object):

    """
    A mixin for Nodes that represent C expression statements, which are expressions
    followed by a semicolon. For example, the lines:

        * i = 0;
        * j = a[i] + 8;
        * int a = 3;
        * foo(b)

    are all expression statements.

    Notes
    -----
    An ExprStmt does *not* have children Nodes.
    """

    pass


class List(Node):

    """A sequence of Nodes."""

    is_List = True

    _traversable = ['body']

    def __init__(self, header=None, body=None, footer=None):
        body = as_tuple(body)
        if len(body) == 1 and all(type(i) == List for i in [self, body[0]]):
            # De-nest Lists
            #
            # Note: to avoid disgusting metaclass voodoo (due to
            # https://stackoverflow.com/questions/56514586/\
            #     arguments-of-new-and-init-for-metaclasses)
            # we change the internal state here in __init__
            # rather than in __new__
            self._args['header'] = self.header = as_tuple(header) + body[0].header
            self._args['body'] = self.body = body[0].body
            self._args['footer'] = self.footer = as_tuple(footer) + body[0].footer
        else:
            self.header = as_tuple(header)
            self.body = as_tuple(body)
            self.footer = as_tuple(footer)

    def __repr__(self):
        return "<%s (%d, %d, %d)>" % (self.__class__.__name__, len(self.header),
                                      len(self.body), len(self.footer))

    @property
    def functions(self):
        return ()

    @property
    def free_symbols(self):
        return ()

    @property
    def defines(self):
        return ()


class Block(List):

    """A sequence of Nodes, wrapped in a block {...}."""

    is_Block = True

    def __init__(self, header=None, body=None, footer=None):
        self.header = as_tuple(header)
        self.body = as_tuple(body)
        self.footer = as_tuple(footer)


class Element(Node):

    """
    A generic node. Can be a comment, a statement, ... or anything that cannot
    be expressed through an IET type.
    """

    is_Element = True

    def __init__(self, element):
        assert isinstance(element, (c.Comment, c.Statement, c.Value, c.Initializer,
                                    c.Pragma, c.Line, c.Assign, c.POD))
        self.element = element

    def __repr__(self):
        return "Element::\n\t%s" % (self.element)


class Call(ExprStmt, Node):

    """
    A function call.

    Parameters
    ----------
    name : str or FunctionFromPointer
        The called function.
    arguments : list of Basic, optional
        The objects in input to the function call.
    retobj : Symbol or Indexed, optional
        The object the return value of the Call is assigned to.
    is_indirect : bool, optional
        If True, the object represents an indirect function call. The emitted
        code will be `name, arg1, ..., argN` rather than `name(arg1, ..., argN)`.
        Defaults to False.
    """

    is_Call = True

    def __init__(self, name, arguments=None, retobj=None, is_indirect=False):
        if isinstance(name, FunctionFromPointer):
            self.base = name.base
        else:
            self.base = None
        self.name = str(name)
        self.arguments = as_tuple(arguments)
        self.retobj = retobj
        self.is_indirect = is_indirect

    def __repr__(self):
        ret = "" if self.retobj is None else "%s = " % self.retobj
        return "%sCall::\n\t%s(...)" % (ret, self.name)

    @property
    def functions(self):
        retval = [i.function for i in self.arguments
                  if isinstance(i, (AbstractFunction, Indexed, LocalObject))]
        if self.base is not None:
            retval.append(self.base.function)
        if self.retobj is not None:
            retval.append(self.retobj.function)
        return tuple(retval)

    @property
    def children(self):
        return tuple(i for i in self.arguments if isinstance(i, (Call, Lambda)))

    @cached_property
    def free_symbols(self):
        free = set()
        for i in self.arguments:
            if isinstance(i, numbers.Number):
                continue
            elif isinstance(i, AbstractFunction):
                if i.is_ArrayBasic:
                    free.add(i)
                else:
                    # Always passed by _C_name since what actually needs to be
                    # provided is the pointer to the corresponding C struct
                    free.add(i._C_symbol)
            else:
                free.update(i.free_symbols)
        if self.base is not None:
            free.add(self.base)
        if self.retobj is not None:
            free.update(self.retobj.free_symbols)
        return free

    @property
    def defines(self):
        ret = ()
        if self.base is not None:
            ret += (self.base,)
        if self.retobj is not None:
            ret += (self.retobj,)
        return ret


class Expression(ExprStmt, Node):

    """
    A node encapsulating a ClusterizedEq.

    Parameters
    ----------
    expr : ClusterizedEq
        The encapsulated expression.
    pragmas : cgen.Pragma or list of cgen.Pragma, optional
        A bag of pragmas attached to this Expression.
    """

    is_Expression = True

    def __init__(self, expr, pragmas=None):
        self.__expr_finalize__(expr, pragmas)

    def __expr_finalize__(self, expr, pragmas):
        """Finalize the Expression initialization."""
        self._expr = expr
        self._pragmas = as_tuple(pragmas)

    def __repr__(self):
        return "<%s::%s>" % (self.__class__.__name__,
                             filter_ordered([f.func for f in self.functions]))

    @property
    def expr(self):
        return self._expr

    @property
    def pragmas(self):
        return self._pragmas

    @property
    def dtype(self):
        return self.expr.dtype

    @property
    def output(self):
        """The Symbol/Indexed this Expression writes to."""
        return self.expr.lhs

    @cached_property
    def reads(self):
        """The Functions read by the Expression."""
        return detect_io(self.expr, relax=True)[0]

    @cached_property
    def write(self):
        """The Function written by the Expression."""
        return self.expr.lhs.base.function

    @cached_property
    def dimensions(self):
        retval = flatten(i.indices for i in self.functions if i.is_Indexed)
        return tuple(filter_ordered(retval))

    @property
    def is_scalar(self):
        """True if a scalar expression, False otherwise."""
        return self.expr.lhs.is_Symbol

    @property
    def is_tensor(self):
        """True if a tensor expression, False otherwise."""
        return not self.is_scalar

    @property
    def is_definition(self):
        """
        True if it is an assignment, False otherwise
        """
        return ((self.is_scalar and not self.is_Increment) or
                (self.is_tensor and isinstance(self.expr.rhs, ListInitializer)))

    @property
    def defines(self):
        return (self.write,) if self.is_definition else ()

    @property
    def free_symbols(self):
        return tuple(self.expr.free_symbols)

    @cached_property
    def functions(self):
        functions = list(self.reads)
        if self.write is not None:
            functions.append(self.write)
        return tuple(filter_ordered(functions))


class AugmentedExpression(Expression):

    """A node representing an augmented assignment, such as +=, -=, &=, ...."""

    is_Increment = True

    def __init__(self, expr, op, pragmas=None):
        super(AugmentedExpression, self).__init__(expr, pragmas=pragmas)
        self.op = op


class Increment(AugmentedExpression):

    """Shortcut for ``AugmentedExpression(expr, '+'), since it's so widely used."""

    def __init__(self, expr, pragmas=None):
        super(Increment, self).__init__(expr, '+', pragmas=pragmas)


class Iteration(Node):

    """
    Implement a for-loop over nodes.

    Parameters
    ----------
    nodes : Node or list of Node
        The for-loop body.
    dimension : Dimension
        The Dimension over which the for-loop iterates.
    limits : expr-like or 3-tuple
        If an expression, it represents the for-loop max point; in this case, the
        min point is 0 and the step increment is unitary. If a 3-tuple, the
        format is ``(min point, max point, stepping)``.
    direction: IterationDirection, optional
        The for-loop direction. Accepted:
        - ``Forward``: i += stepping (defaults)
        - ``Backward``: i -= stepping
    properties : Property or list of Property, optional
        Iteration decorators, denoting properties such as parallelism.
    pragmas : cgen.Pragma or list of cgen.Pragma, optional
        A bag of pragmas attached to this Iteration.
    uindices : DerivedDimension or list of DerivedDimension, optional
        An uindex is an additional iteration variable defined by the for-loop. The
        for-loop bounds are independent of all ``uindices`` (hence the name uindex,
        or "unbounded index"). An uindex must have ``dimension`` as its parent.
    """

    is_Iteration = True

    _traversable = ['nodes']

    def __init__(self, nodes, dimension, limits, direction=None, properties=None,
                 pragmas=None, uindices=None):
        self.nodes = as_tuple(nodes)
        self.dim = dimension
        self.index = self.dim.name
        self.direction = direction or Forward

        # Generate loop limits
        if isinstance(limits, Iterable):
            assert(len(limits) == 3)
            self.limits = tuple(limits)
        elif self.dim.is_Incr:
            self.limits = (self.dim.symbolic_min, limits, self.dim.step)
        else:
            self.limits = (0, limits, 1)

        # Track this Iteration's properties, pragmas and unbounded indices
        properties = as_tuple(properties)
        assert (i in Property._KNOWN for i in properties)
        self.properties = as_tuple(filter_sorted(properties))
        self.pragmas = as_tuple(pragmas)
        self.uindices = as_tuple(uindices)
        assert all(i.is_Derived and self.dim in i._defines for i in self.uindices)

    def __repr__(self):
        properties = ""
        if self.properties:
            properties = [str(i) for i in self.properties]
            properties = "WithProperties[%s]::" % ",".join(properties)
        index = self.index
        if self.uindices:
            index += '[%s]' % ','.join(i.name for i in self.uindices)
        return "<%sIteration %s; %s>" % (properties, index, self.limits)

    @property
    def is_Affine(self):
        return AFFINE in self.properties

    @property
    def is_Sequential(self):
        return SEQUENTIAL in self.properties

    @property
    def is_Parallel(self):
        return PARALLEL in self.properties

    @property
    def is_ParallelAtomic(self):
        return PARALLEL_IF_ATOMIC in self.properties

    @property
    def is_ParallelPrivate(self):
        return PARALLEL_IF_PVT in self.properties

    @property
    def is_ParallelRelaxed(self):
        return any([self.is_Parallel, self.is_ParallelAtomic, self.is_ParallelPrivate])

    @property
    def is_Vectorized(self):
        return VECTORIZED in self.properties

    @property
    def ncollapsed(self):
        for i in self.properties:
            if i.name == 'collapsed':
                return i.val
        return 0

    @property
    def symbolic_bounds(self):
        """A 2-tuple representing the symbolic bounds [min, max] of the Iteration."""
        _min = self.limits[0]
        _max = self.limits[1]
        try:
            _min = as_symbol(_min)
        except TypeError:
            # A symbolic expression
            pass
        try:
            _max = as_symbol(_max)
        except TypeError:
            # A symbolic expression
            pass
        return (_min, _max)

    @property
    def symbolic_size(self):
        """The symbolic size of the Iteration."""
        return self.symbolic_bounds[1] - self.symbolic_bounds[0] + 1

    @property
    def symbolic_min(self):
        """The symbolic min of the Iteration."""
        return self.symbolic_bounds[0]

    @property
    def symbolic_max(self):
        """The symbolic max of the Iteration."""
        return self.symbolic_bounds[1]

    def bounds(self, _min=None, _max=None):
        """
        The bounds [min, max] of the Iteration, as numbers if min/max are supplied,
        as symbols otherwise.
        """
        _min = _min if _min is not None else self.limits[0]
        _max = _max if _max is not None else self.limits[1]

        return (_min, _max)

    @property
    def step(self):
        """The step value."""
        return self.limits[2]

    def size(self, _min=None, _max=None):
        """The size of the iteration space if _min/_max are supplied, None otherwise."""
        _min, _max = self.bounds(_min, _max)
        return _max - _min + 1

    @property
    def functions(self):
        """All Functions appearing in the Iteration header."""
        return ()

    @property
    def free_symbols(self):
        """All Symbols appearing in the Iteration header."""
        return tuple(self.symbolic_min.free_symbols) \
            + tuple(self.symbolic_max.free_symbols) \
            + self.uindices \
            + tuple(flatten(i.symbolic_min.free_symbols for i in self.uindices)) \
            + tuple(flatten(i.symbolic_incr.free_symbols for i in self.uindices))

    @property
    def defines(self):
        """All Symbols defined in the Iteration header."""
        return self.dimensions

    @property
    def dimensions(self):
        """All Dimensions appearing in the Iteration header."""
        return tuple(self.dim._defines) + self.uindices

    @property
    def write(self):
        """All Functions written to in this Iteration"""
        return []


class While(Node):

    """
    Implement a while-loop.

    Parameters
    ----------
    condition : sympy.Function or sympy.Relation or bool
        The while-loop exit condition.
    body : Node or list of Node, optional
        The whie-loop body.
    """

    is_While = True

    _traversable = ['body']

    def __init__(self, condition, body=None):
        self.condition = condition
        self.body = as_tuple(body)

    def __repr__(self):
        return "<While %s; %d>" % (self.condition, len(self.body))


class Callable(Node):

    """
    A callable function.

    Parameters
    ----------
    name : str
        The name of the callable.
    body : Node or list of Node
        The Callable body.
    retval : str
        The return type of Callable.
    parameters : list of Basic, optional
        The objects in input to the Callable.
    prefix : list of str, optional
        Qualifiers to prepend to the Callable signature. Defaults to ``('static',
        'inline')``.
    """

    is_Callable = True

    _traversable = ['body']

    def __init__(self, name, body, retval, parameters=None, prefix=('static', 'inline')):
        self.name = name
        self.body = as_tuple(body)
        self.retval = retval
        self.prefix = as_tuple(prefix)
        self.parameters = as_tuple(parameters)

    def __repr__(self):
        return "%s[%s]<%s; %s>" % (self.__class__.__name__, self.name, self.retval,
                                   ",".join([i._C_typename for i in self.parameters]))

    @property
    def functions(self):
        return tuple(i for i in self.parameters if isinstance(i, AbstractFunction))

    @property
    def free_symbols(self):
        return ()

    @property
    def defines(self):
        return ()


class Conditional(Node):

    """
    A node to express if-then-else blocks.

    Parameters
    ----------
    condition : expr-like
        The if condition.
    then_body : Node or list of Node
        The then body.
    else_body : Node or list of Node
        The else body.
    """

    is_Conditional = True

    _traversable = ['then_body', 'else_body']

    def __init__(self, condition, then_body, else_body=None):
        self.condition = condition
        self.then_body = as_tuple(then_body)
        self.else_body = as_tuple(else_body)

    def __repr__(self):
        if self.else_body:
            return "<[%s] ? [%s] : [%s]>" %\
                (ccode(self.condition), repr(self.then_body), repr(self.else_body))
        else:
            return "<[%s] ? [%s]" % (ccode(self.condition), repr(self.then_body))

    @property
    def functions(self):
        ret = []
        for i in self.condition.free_symbols:
            try:
                ret.append(i.function)
            except AttributeError:
                pass
        return tuple(ret)

    @property
    def free_symbols(self):
        return tuple(self.condition.free_symbols)

    @property
    def defines(self):
        return ()


class Break(Node):

    """A node representing a break statement"""

    is_Break = True


# Second level IET nodes

class TimedList(List):

    """
    Wrap a Node with C-level timers.

    Parameters
    ----------
    timer : Timer
        The Timer used by the TimedList.
    lname : str
        A unique name for the timed code block.
    body : Node or list of Node
        The TimedList body.
    """

    def __init__(self, timer, lname, body):
        self._name = lname
        self._timer = timer

        super().__init__(header=c.Line('START_TIMER(%s)' % lname),
                         body=body,
                         footer=c.Line('STOP_TIMER(%s,%s)' % (lname, timer.name)))

    @classmethod
    def _start_timer_header(cls):
        return ('START_TIMER(S)', ('struct timeval start_ ## S , end_ ## S ; '
                                   'gettimeofday(&start_ ## S , NULL);'))

    @classmethod
    def _stop_timer_header(cls):
        return ('STOP_TIMER(S,T)', ('gettimeofday(&end_ ## S, NULL); T->S += (double)'
                                    '(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)'
                                    '(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;'))

    @property
    def name(self):
        return self._name

    @property
    def timer(self):
        return self._timer

    @property
    def free_symbols(self):
        return ()


class PointerCast(ExprStmt, Node):

    """
    A node encapsulating a cast of a raw C pointer to a multi-dimensional array.
    """

    is_PointerCast = True

    def __init__(self, function, obj=None, alignment=True):
        self.function = function
        self.obj = obj
        self.alignment = alignment

    def __repr__(self):
        return "<PointerCast(%s)>" % self.function

    @property
    def castshape(self):
        """
        The shape used in the left-hand side and right-hand side of the PointerCast.
        """
        if self.function.is_ArrayBasic:
            return self.function.symbolic_shape[1:]
        else:
            return tuple(self.function._C_get_field(FULL, d).size
                         for d in self.function.dimensions[1:])

    @property
    def functions(self):
        return (self.function,)

    @property
    def free_symbols(self):
        """
        The symbols required by the PointerCast.

        This may include DiscreteFunctions as well as Dimensions.
        """
        f = self.function
        if f.is_ArrayBasic:
            return tuple(flatten(s.free_symbols for s in f.symbolic_shape[1:]))
        else:
            return ()

    @property
    def defines(self):
        return ()


class Dereference(ExprStmt, Node):

    """
    A node encapsulating a dereference from a `pointer` to a `pointee`.
    The following cases are supported:

        * `pointer` is a PointerArray and `pointee` is an Array (typical case).
        * `pointer` is an ArrayObject representing a pointer to a C struct while
          `pointee` is a field in `pointer`.
    """

    is_Dereference = True

    def __init__(self, pointee, pointer):
        self.pointee = pointee
        self.pointer = pointer

    def __repr__(self):
        return "<Dereference(%s,%s)>" % (self.pointee, self.pointer)

    @property
    def functions(self):
        return (self.pointee, self.pointer)

    @property
    def free_symbols(self):
        return ((self.pointee.indexed.label, self.pointer.indexed.label) +
                tuple(flatten(i.free_symbols for i in self.pointee.symbolic_shape[1:])) +
                tuple(self.pointer.free_symbols))

    @property
    def defines(self):
        return (self.pointee,)


class LocalExpression(Expression):

    """
    A node encapsulating a SymPy equation which also defines its LHS.
    """

    is_LocalExpression = True

    @cached_property
    def write(self):
        return self.expr.lhs.function

    @property
    def defines(self):
        return (self.write, )


class ForeignExpression(Expression):

    """A node representing a SymPy FunctionFromPointer expression."""

    is_ForeignExpression = True

    def __init__(self, expr, dtype, **kwargs):
        self._dtype = dtype
        self._is_increment = kwargs.get('is_Increment', False)
        self.__expr_finalize__(expr)

    @property
    def dtype(self):
        return self._dtype

    @property
    def output(self):
        return self.expr.base

    @property
    def write(self):
        if isinstance(self.output, (Symbol, Indexed)):
            return self.output.function
        else:
            return None

    @property
    def is_Increment(self):
        return self._is_increment

    @property
    def is_scalar(self):
        return False

    @property
    def is_tensor(self):
        return False


class Lambda(Node):

    """
    A callable C++ lambda function. Several syntaxes are possible; here we
    implement one of the common ones:

        [captures](parameters){body}

    For more info about C++ lambda functions:

        https://en.cppreference.com/w/cpp/language/lambda

    Parameters
    ----------
    body : Node or list of Node
        The lambda function body.
    captures : list of str or expr-like, optional
        The captures of the lambda function.
    parameters : list of Basic or expr-like, optional
        The objects in input to the lambda function.
    """

    is_Lambda = True

    _traversable = ['body']

    def __init__(self, body, captures=None, parameters=None):
        self.body = as_tuple(body)
        self.captures = as_tuple(captures)
        self.parameters = as_tuple(parameters)

    def __repr__(self):
        return "Lambda[%s](%s)" % (self.captures, self.parameters)

    @cached_property
    def free_symbols(self):
        return set(self.parameters)

    @property
    def defines(self):
        return ()


class Section(List):

    """
    A sequence of nodes.

    Functionally, a Section is identical to a List; that is,
    they generate the same code (i.e., their ``body``). However, a Section should
    be used to define sub-trees that, for some reasons, have a relevance within
    the IET (e.g., groups of statements that logically represent the same
    computation unit).
    """

    is_Section = True

    def __init__(self, name, body=None, is_subsection=False):
        super(Section, self).__init__(body=body)
        self.name = name
        self.is_subsection = is_subsection

    def __repr__(self):
        return "<Section (%s)>" % self.name

    @property
    def roots(self):
        return self.body


class ExpressionBundle(List):

    """
    A sequence of Expressions.
    """

    is_ExpressionBundle = True

    def __init__(self, ispace, ops, traffic, body=None):
        super(ExpressionBundle, self).__init__(body=body)
        self.ispace = ispace
        self.ops = ops
        self.traffic = traffic

    def __repr__(self):
        return "<ExpressionBundle (%d)>" % len(self.exprs)

    @property
    def exprs(self):
        return self.body

    @property
    def size(self):
        return self.ispace.size


class Prodder(Call):

    """
    A Call promoting asynchronous progress, to minimize latency.

    Example use cases:

        * To trigger asynchronous progress in the case of distributed-memory
          parallelism.
        * Software prefetching.
    """

    def __init__(self, name, arguments=None, single_thread=False, periodic=False):
        super(Prodder, self).__init__(name, arguments)

        # Prodder properties
        self._single_thread = single_thread
        self._periodic = periodic

    @property
    def single_thread(self):
        return self._single_thread

    @property
    def periodic(self):
        return self._periodic


class PragmaList(List):

    """
    A floating sequence of pragmas.
    """

    def __init__(self, pragmas, functions=None, **kwargs):
        super().__init__(header=pragmas)
        self._functions = as_tuple(functions)

    @property
    def pragmas(self):
        return self.header

    @property
    def functions(self):
        return self._functions

    @property
    def free_symbols(self):
        return self._functions


class ParallelIteration(Iteration):

    """
    Implement a parallel for-loop.
    """

    is_ParallelIteration = True

    def __init__(self, *args, **kwargs):
        pragmas, kwargs, properties = self._make_header(**kwargs)
        super().__init__(*args, pragmas=pragmas, properties=properties, **kwargs)

    @classmethod
    def _make_header(cls, **kwargs):
        construct = cls._make_construct(**kwargs)
        clauses = cls._make_clauses(**kwargs)
        header = c.Pragma(' '.join([construct] + clauses))

        # Extract the Iteration Properties
        properties = cls._process_properties(**kwargs)

        # Drop the unrecognised or unused kwargs
        kwargs = cls._process_kwargs(**kwargs)

        return (header,), kwargs, properties

    @classmethod
    def _make_construct(cls, **kwargs):
        # To be overridden by subclasses
        raise NotImplementedError

    @classmethod
    def _make_clauses(cls, **kwargs):
        return []

    @classmethod
    def _process_properties(cls, **kwargs):
        properties = as_tuple(kwargs.get('properties'))
        properties += (COLLAPSED(kwargs.get('ncollapse', 1)),)

        return properties

    @classmethod
    def _process_kwargs(cls, **kwargs):
        kwargs.pop('pragmas', None)
        kwargs.pop('properties', None)

        # Recognised clauses
        kwargs.pop('ncollapse', None)
        kwargs.pop('reduction', None)

        return kwargs

    @cached_property
    def collapsed(self):
        ret = [self]
        for i in range(self.ncollapsed - 1):
            ret.append(ret[i].nodes[0])
        assert all(i.is_Iteration for i in ret)
        return tuple(ret)


class ParallelTree(List):

    """
    This class is to group together a parallel for-loop with some setup
    statements, for example:

        .. code-block:: C

          int chunk_size = ...
          #pragma parallel for ... schedule(..., chunk_size)
          for (int i = ...)
          {
            ...
          }
    """

    _traversable = ['prefix', 'body']

    def __init__(self, prefix, body, nthreads=None):
        # Normalize and sanity-check input
        body = as_tuple(body)
        assert len(body) == 1 and body[0].is_Iteration

        self.prefix = as_tuple(prefix)
        self.nthreads = nthreads

        super().__init__(body=body)

    def __getattr__(self, name):
        if 'body' in self.__dict__:
            # During unpickling, `__setattr__` calls `__getattr__(..., 'body')`,
            # which would cause infinite recursion if we didn't check whether
            # 'body' is present or not
            return getattr(self.body[0], name)
        raise AttributeError

    @property
    def functions(self):
        return as_tuple(self.nthreads)

    @property
    def root(self):
        return self.body[0]


class ParallelBlock(Block):

    """
    A sequence of Nodes, wrapped in a parallel block {...}.
    """

    is_ParallelBlock = True

    def __init__(self, body, private=None):
        # Normalize and sanity-check input. A bit ugly, but it makes everything
        # much simpler to manage and reconstruct
        body = as_tuple(body)
        assert len(body) == 1
        body = body[0]
        assert body.is_List
        if isinstance(body, ParallelTree):
            partree = body
        elif body.is_List:
            assert len(body.body) == 1 and isinstance(body.body[0], ParallelTree)
            assert len(body.footer) == 0
            partree = body.body[0]
            partree = partree._rebuild(prefix=(List(header=body.header,
                                                    body=partree.prefix)))

        header = self._make_header(partree.nthreads, private)
        super().__init__(header=header, body=partree)

    @classmethod
    def _make_header(cls, nthreads, private=None):
        return None

    @property
    def partree(self):
        return self.body[0]

    @property
    def root(self):
        return self.partree.root

    @property
    def nthreads(self):
        return self.partree.nthreads

    @property
    def collapsed(self):
        return self.partree.collapsed


class SyncSpot(List):

    """
    A node representing one or more synchronization operations, e.g., WaitLock,
    withLock, etc.
    """

    is_SyncSpot = True

    def __init__(self, sync_ops, body=None):
        super().__init__(body=body)
        self.sync_ops = sync_ops

    def __repr__(self):
        return "<SyncSpot (%s)>" % ",".join(str(i) for i in self.sync_ops)


class CBlankLine(List):

    def __init__(self, **kwargs):
        super().__init__(header=c.Line())

    def __repr__(self):
        return ""


def DummyExpr(*args):
    return Expression(DummyEq(*args))


BlankLine = CBlankLine()
Return = lambda i='': Element(c.Statement('return%s' % ((' %s' % i) if i else i)))


# Nodes required for distributed-memory halo exchange


class HaloSpot(Node):

    """
    A halo exchange operation (e.g., send, recv, wait, ...) required to
    correctly execute the subtree in the case of distributed-memory parallelism.
    """

    is_HaloSpot = True

    _traversable = ['body']

    def __init__(self, halo_scheme, body=None):
        super(HaloSpot, self).__init__()
        self._halo_scheme = halo_scheme
        if isinstance(body, Node):
            self._body = body
        elif isinstance(body, (list, tuple)) and len(body) == 1:
            self._body = body[0]
        elif body is None:
            self._body = List()
        else:
            raise ValueError("`body` is expected to be a single Node")

    def __repr__(self):
        functions = "(%s)" % ",".join(i.name for i in self.functions)
        return "<%s%s>" % (self.__class__.__name__, functions)

    @property
    def halo_scheme(self):
        return self._halo_scheme

    @property
    def fmapper(self):
        return self.halo_scheme.fmapper

    @property
    def omapper(self):
        return self.halo_scheme.omapper

    @property
    def dimensions(self):
        return self.halo_scheme.dimensions

    @property
    def arguments(self):
        return self.halo_scheme.arguments

    @property
    def is_empty(self):
        return len(self.halo_scheme) == 0

    @property
    def body(self):
        return self._body

    @property
    def functions(self):
        return tuple(self.fmapper)

    @property
    def free_symbols(self):
        return ()

    @property
    def defines(self):
        return ()


# Utility classes


class IterationTree(tuple):

    """
    Represent a sequence of nested Iterations.
    """

    @property
    def root(self):
        return self[0] if self else None

    @property
    def inner(self):
        return self[-1] if self else None

    @property
    def dimensions(self):
        return [i.dim for i in self]

    def __repr__(self):
        return "IterationTree%s" % super(IterationTree, self).__repr__()

    def __getitem__(self, key):
        ret = super(IterationTree, self).__getitem__(key)
        return IterationTree(ret) if isinstance(key, slice) else ret


MetaCall = namedtuple('MetaCall', 'root local')
"""
Metadata for Callables. ``root`` is a pointer to the callable
Iteration/Expression tree. ``local`` is a boolean indicating whether the
definition of the callable is known or not.
"""
