"""
Extended SymPy hierarchy.
"""

import numpy as np
import sympy
from sympy import Expr, Integer, Function, Symbol

from devito.symbolics.printer import ccode
from devito.tools import Pickable, as_tuple, is_integer

__all__ = ['CondEq', 'CondNe', 'IntDiv', 'FunctionFromPointer', 'FieldFromPointer',
           'FieldFromComposite', 'ListInitializer', 'Byref', 'IndexedPointer',
           'DefFunction', 'InlineIf', 'Macro', 'Literal', 'INT', 'FLOAT', 'DOUBLE',
           'FLOOR', 'cast_mapper']


class CondEq(sympy.Eq):

    """
    A customized version of sympy.Eq representing a conditional equality.
    It suppresses evaluation.
    """

    def __new__(cls, *args, **kwargs):
        return sympy.Eq.__new__(cls, *args, evaluate=False)

    @property
    def canonical(self):
        return self

    @property
    def negated(self):
        return CondNe(*self.args, evaluate=False)


class CondNe(sympy.Ne):

    """
    A customized version of sympy.Ne representing a conditional inequality.
    It suppresses evaluation.
    """

    def __new__(cls, *args, **kwargs):
        return sympy.Ne.__new__(cls, *args, evaluate=False)

    @property
    def canonical(self):
        return self

    @property
    def negated(self):
        return CondEq(*self.args, evaluate=False)


class IntDiv(sympy.Expr):

    """
    A support type for integer division. Should only be used by the compiler
    for code generation purposes (i.e., not for symbolic manipulation).
    This works around the annoying way SymPy represents integer division,
    namely as a ``Mul`` between the numerator and the reciprocal of the
    denominator (e.g., ``a*3.args -> (a, 1/3)), which ends up generating
    "weird" C code.
    """

    is_Atom = True

    def __new__(cls, lhs, rhs, params=None):
        try:
            rhs = Integer(rhs)
            if rhs == 0:
                raise ValueError("Cannot divide by 0")
            elif rhs == 1:
                return lhs
        except TypeError:
            # We must be sure the symbolic RHS is of type int
            if not hasattr(rhs, 'dtype'):
                raise ValueError("Symbolic RHS `%s` lacks dtype" % rhs)
            if not issubclass(rhs.dtype, np.integer):
                raise ValueError("Symbolic RHS `%s` must be of type `int`, found "
                                 "`%s` instead" % (rhs, rhs.dtype))
        obj = sympy.Expr.__new__(cls, lhs, rhs)
        obj.lhs = lhs
        obj.rhs = rhs
        return obj

    def __str__(self):
        return "%s / %s" % (self.lhs, self.rhs)

    __repr__ = __str__


class FunctionFromPointer(sympy.Expr, Pickable):

    """
    Symbolic representation of the C notation ``pointer->function(params)``.
    """

    def __new__(cls, function, pointer, params=None):
        args = []
        if isinstance(pointer, str):
            pointer = Symbol(pointer)
        args.append(pointer)
        if isinstance(function, (DefFunction, FunctionFromPointer)):
            args.append(function)
        elif not isinstance(function, str):
            raise ValueError("`function` must be FunctionFromPointer or str")
        _params = []
        for p in as_tuple(params):
            if isinstance(p, str):
                _params.append(Symbol(p))
            elif not isinstance(p, Expr):
                raise ValueError("`params` must be an iterable of Expr or str")
            else:
                _params.append(p)
        args.extend(_params)
        obj = sympy.Expr.__new__(cls, *args)
        obj.function = function
        obj.pointer = pointer
        obj.params = tuple(_params)
        return obj

    def __str__(self):
        return '%s->%s(%s)' % (self.pointer, self.function,
                               ", ".join(str(i) for i in as_tuple(self.params)))

    __repr__ = __str__

    def _hashable_content(self):
        return super(FunctionFromPointer, self)._hashable_content() +\
            (self.function, self.pointer) + self.params

    @property
    def base(self):
        if isinstance(self.pointer, FunctionFromPointer):
            # FunctionFromPointer may be nested
            return self.pointer.base
        else:
            return self.pointer

    @property
    def dtype(self):
        return self.base.function.dtype

    # Pickling support
    _pickle_args = ['function', 'pointer']
    _pickle_kwargs = ['params']
    __reduce_ex__ = Pickable.__reduce_ex__


class FieldFromPointer(FunctionFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``pointer->field``.
    """

    def __new__(cls, field, pointer):
        return FunctionFromPointer.__new__(cls, field, pointer)

    def __str__(self):
        return '%s->%s' % (self.pointer, self.field)

    @property
    def field(self):
        return self.function

    # Our __new__ cannot accept the params argument
    _pickle_kwargs = []

    __repr__ = __str__


class FieldFromComposite(FunctionFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``composite.field``,
    where ``composite`` is a struct/union/...
    """

    def __new__(cls, field, composite):
        return FunctionFromPointer.__new__(cls, field, composite)

    def __str__(self):
        return '%s.%s' % (self.composite, self.field)

    @property
    def field(self):
        return self.function

    @property
    def composite(self):
        return self.pointer

    # Our __new__ cannot accept the params argument
    _pickle_kwargs = []

    __repr__ = __str__


class ListInitializer(sympy.Expr, Pickable):

    """
    Symbolic representation of the C++ list initializer notation ``{a, b, ...}``.
    """

    def __new__(cls, params):
        args = []
        for p in as_tuple(params):
            if isinstance(p, str):
                args.append(Symbol(p))
            elif is_integer(p):
                args.append(Integer(p))
            elif not isinstance(p, Expr):
                raise ValueError("`params` must be an iterable of Expr or str")
            else:
                args.append(p)
        obj = sympy.Expr.__new__(cls, *args)
        obj.params = tuple(args)
        return obj

    def __str__(self):
        return "{%s}" % ", ".join(str(i) for i in self.params)

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['params']
    __reduce_ex__ = Pickable.__reduce_ex__


class Byref(sympy.Expr, Pickable):

    """
    Symbolic representation of the C notation ``&symbol``.
    """

    def __new__(cls, base):
        if isinstance(base, str):
            base = Symbol(base)
        elif not isinstance(base, sympy.Expr):
            raise ValueError("`base` must be sympy.Expr or str")
        obj = sympy.Expr.__new__(cls, base)
        obj._base = base
        return obj

    @property
    def base(self):
        return self._base

    def __str__(self):
        if self.base.is_Symbol:
            return "&%s" % ccode(self.base)
        else:
            return "&(%s)" % ccode(self.base)

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['base']
    __reduce_ex__ = Pickable.__reduce_ex__


class IndexedPointer(sympy.Expr, Pickable):

    """
    Symbolic representation of the C notation ``symbol[...]``

    Unlike a sympy.Indexed, an IndexedPointer accepts, as base, objects that
    are not necessarily a Symbol or an IndexedBase, such as a FieldFromPointer.
    """

    def __new__(cls, base, index):
        if isinstance(base, (str, sympy.IndexedBase, sympy.Symbol)):
            return sympy.Indexed(base, index)
        elif not isinstance(base, sympy.Basic):
            raise ValueError("`base` must be of type sympy.Basic")
        obj = sympy.Expr.__new__(cls, base)
        obj._base = base
        obj._index = as_tuple(index)
        return obj

    @property
    def base(self):
        return self._base

    @property
    def index(self):
        return self._index

    def __str__(self):
        return "%s%s" % (self.base, ''.join('[%s]' % i for i in self.index))

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['base', 'index']
    __reduce_ex__ = Pickable.__reduce_ex__


class DefFunction(Function, Pickable):

    """
    A definitely-defined sympy.Function, to work around:

        https://github.com/sympy/sympy/issues/4297
    """

    def __new__(cls, name, arguments=None):
        arguments = as_tuple(arguments)
        obj = Function.__new__(cls, name, *arguments)
        obj._name = name
        obj._arguments = arguments
        return obj

    @property
    def name(self):
        return self._name

    @property
    def arguments(self):
        return self._arguments

    def __str__(self):
        return "%s(%s)" % (self.name, ', '.join(str(i) for i in self.arguments))

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['name']
    _pickle_kwargs = ['arguments']
    __reduce_ex__ = Pickable.__reduce_ex__


class InlineIf(sympy.Expr, Pickable):

    """
    Symbolic representation of the C notation ``(C) ? a : b``.
    """

    def __new__(cls, cond, true_expr, false_expr):
        if not isinstance(cond, sympy.core.relational.Relational):
            raise ValueError("`cond` must be of type sympy.relational.Relational")
        if not isinstance(true_expr, sympy.Expr):
            raise ValueError("`true_expr` must be of type sympy.Expr")
        if not isinstance(false_expr, sympy.Expr):
            raise ValueError("`false_expr` must be of type sympy.Expr")
        obj = sympy.Expr.__new__(cls, cond, true_expr, false_expr)
        obj._cond = cond
        obj._true_expr = true_expr
        obj._false_expr = false_expr
        return obj

    @property
    def cond(self):
        return self._cond

    @property
    def true_expr(self):
        return self._true_expr

    @property
    def false_expr(self):
        return self._false_expr

    def __str__(self):
        return "(%s) ? %s : %s" % (self.cond, self.true_expr, self.false_expr)

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['cond', 'true_expr', 'false_expr']
    __reduce_ex__ = Pickable.__reduce_ex__


class Macro(sympy.Symbol):

    """
    Symbolic representation of a C macro.
    """
    pass


class Literal(sympy.Symbol):

    """
    Symbolic representation of a Literal element.
    """
    pass


INT = Function('INT')
FLOAT = Function('FLOAT')
DOUBLE = Function('DOUBLE')
FLOOR = Function('floor')

cast_mapper = {np.float32: FLOAT, float: DOUBLE, np.float64: DOUBLE}
