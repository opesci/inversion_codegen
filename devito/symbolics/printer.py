"""
Utilities to turn SymPy objects into C strings.
"""

import numpy as np

from mpmath.libmp import prec_to_dps, to_str
from sympy.printing.precedence import precedence
from sympy.printing.c import C99CodePrinter


__all__ = ['ccode']


class CodePrinter(C99CodePrinter):

    custom_functions = {'INT': '(int)', 'FLOAT': '(float)', 'DOUBLE': '(double)'}

    """
    Decorator for sympy.printing.ccode.CCodePrinter.

    Parameters
    ----------
    settings : dict
        Options for code printing.
    """
    def __init__(self, dtype=np.float32, settings={}):
        self.dtype = dtype
        C99CodePrinter.__init__(self, settings)
        self.known_functions.update(self.custom_functions)

    def _print_Function(self, expr):
        # There exist no unknown Functions
        if expr.func.__name__ not in self.known_functions:
            self.known_functions[expr.func.__name__] = expr.func.__name__
        return super(CodePrinter, self)._print_Function(expr)

    def _print_CondEq(self, expr):
        return "%s == %s" % (self._print(expr.lhs), self._print(expr.rhs))

    def _print_Indexed(self, expr):
        """
        Print an Indexed as a C-like multidimensional array.

        Examples
        --------
        U[t,x,y,z] -> U[t][x][y][z]
        """
        output = self._print(expr.base.label) \
            + ''.join(['[' + self._print(x) + ']' for x in expr.indices])

        return output

    def _print_Rational(self, expr):
        """Print a Rational as a C-like float/float division."""
        # This method and _print_Float below forcefully add a F to any
        # literals generated in code. This forces all float literals
        # to be 32-bit floats.
        # http://en.cppreference.com/w/cpp/language/floating_literal
        p, q = int(expr.p), int(expr.q)
        if self.dtype == np.float64:
            return '%d.0/%d.0' % (p, q)
        else:
            return '%d.0F/%d.0F' % (p, q)

    def _print_Pow(self, expr):
        # Need to override because of issue #1627
        # E.g., (Pow(h_x, -1) AND h_x.dtype == np.float32) => 1.0F/h_x
        try:
            if expr.exp == -1 and self.dtype == np.float32:
                PREC = precedence(expr)
                return '1.0F/%s' % self.parenthesize(expr.base, PREC)
        except AttributeError:
            pass
        return super()._print_Pow(expr)

    def _print_Mod(self, expr):
        """Print a Mod as a C-like %-based operation."""
        args = map(ccode, expr.args)
        args = ['('+x+')' for x in args]

        result = '%'.join(args)
        return result

    def _print_Float(self, expr):
        """Print a Float in C-like scientific notation."""
        prec = expr._prec

        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec)
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1

        rv = to_str(expr._mpf_, dps, strip_zeros=strip, max_fixed=-2, min_fixed=2)

        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]

        if self.dtype == np.float32:
            rv = rv + 'F'

        return rv

    def _print_Differentiable(self, expr):
        return "(" + self._print(expr._expr) + ")"

    def _print_FunctionFromPointer(self, expr):
        indices = [self._print(i) for i in expr.params]
        return "%s->%s(%s)" % (expr.pointer, expr.function, ', '.join(indices))

    def _print_FieldFromPointer(self, expr):
        return "%s->%s" % (expr.pointer, expr.field)

    def _print_FieldFromComposite(self, expr):
        return "%s.%s" % (expr.pointer, expr.field)

    def _print_ListInitializer(self, expr):
        return "{%s}" % ', '.join([self._print(i) for i in expr.params])

    def _print_IntDiv(self, expr):
        return expr.__str__()

    _print_Byref = _print_IntDiv
    _print_IndexedPointer = _print_IntDiv
    _print_DefFunction = _print_IntDiv
    _print_InlineIf = _print_IntDiv

    def _print_TrigonometricFunction(self, expr):
        func_name = str(expr.func)
        if self.dtype == np.float32:
            func_name += 'f'
        return func_name + '(' + self._print(*expr.args) + ')'

    def _print_Basic(self, expr):
        return str(expr)


def ccode(expr, dtype=np.float32, **settings):
    """Generate C++ code from an expression.

    Parameters
    ----------
    expr : expr-like
        The expression to be printed.
    settings : dict
        Options for code printing.

    Returns
    -------
    str
        The resulting code as a C++ string. If something went south, returns
        the input ``expr`` itself.
    """
    return CodePrinter(dtype=dtype, settings=settings).doprint(expr, None)
