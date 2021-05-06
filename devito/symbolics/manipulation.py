from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from functools import singledispatch

import sympy
from sympy import Number, Indexed, Symbol, LM, LC
from sympy.core.add import _addsort
from sympy.core.mul import _mulsort

from devito.symbolics.search import retrieve_indexed, retrieve_functions
from devito.tools import as_list, as_tuple, flatten, split
from devito.types.equation import Eq

__all__ = ['xreplace_indices', 'pow_to_mul', 'as_symbol', 'indexify',
           'split_affine', 'subs_op_args', 'uxreplace', 'aligned_indices',
           'Uxmapper', 'reuse_if_untouched']


def uxreplace(expr, rule):
    """
    An alternative to SymPy's ``xreplace`` for when the caller can guarantee
    that no re-evaluations are necessary or when re-evaluations should indeed
    be avoided at all costs (e.g., to prevent SymPy from unpicking Devito
    transformations, such as factorization).

    The canonical ordering of the arguments is however guaranteed; where this is
    not possible, a re-evaluation will be enforced.

    By avoiding re-evaluations, this function is typically much quicker than
    SymPy's xreplace.

    A further feature of ``uxreplace`` consists of enabling the substitution
    of compound nodes. Consider the expression `a*b*c*d`; if one wants to replace
    `b*c` with say `e*f`, then the following mapper may be passed:
    `{a*b*c*d: {b: e*f, c: None}}`. This way, only the `b` and `c` pertaining to
    `a*b*c*d` will be affected, and in particular `c` will be dropped, while `b`
    will be replaced by `e*f`, thus obtaining `a*d*e*f`.
    """
    return _uxreplace(expr, rule)[0]


def _uxreplace(expr, rule):
    if expr in rule:
        v = rule[expr]
        if not isinstance(v, dict):
            return v, True
        args, eargs = split(expr.args, lambda i: i in v)
        args = [v[i] for i in args if v[i] is not None]
        changed = True
    else:
        args, eargs = [], expr.args
        changed = False

    if rule:
        for a in eargs:
            try:
                ax, flag = _uxreplace(a, rule)
                args.append(ax)
                changed |= flag
            except AttributeError:
                # E.g., un-sympified numbers
                args.append(a)
        if changed:
            return _uxreplace_handle(expr, args), True

    return expr, False


@singledispatch
def _uxreplace_handle(expr, args):
    return expr.func(*args)


@_uxreplace_handle.register(sympy.Add)
def _(expr, args):
    if all(i.is_commutative for i in args):
        _addsort(args)
        _eval_numbers(expr, args)
        return expr.func(*args, evaluate=False)
    else:
        return expr._new_rawargs(*args)


@_uxreplace_handle.register(sympy.Mul)
def _(expr, args):
    if all(i.is_commutative for i in args):
        _mulsort(args)
        _eval_numbers(expr, args)
        return expr.func(*args, evaluate=False)
    else:
        return expr._new_rawargs(*args)


@_uxreplace_handle.register(Eq)
def _(expr, args):
    # Preserve properties such as `implicit_dims`
    return expr.func(*args, subdomain=expr.subdomain, coefficients=expr.substitutions,
                     implicit_dims=expr.implicit_dims)


def _eval_numbers(expr, args):
    """
    Helper function for in-place reduction of the expr arguments.
    """
    numbers, others = split(args, lambda i: i.is_Number)
    if len(numbers) > 1:
        args[:] = [expr.func(*numbers)] + others


class Uxmapper(dict):

    """
    A helper mapper for uxreplace. Any dict can be passed as subs to uxreplace,
    but with a Uxmapper it is also possible to easily express compound replacements,
    such as sub-expressions within n-ary operations (e.g., `a*b` inside the 4-way
    Mul `a*c*b*d`).
    """

    class Uxsubmap(dict):
        @property
        def free_symbols(self):
            return {v for v in self.values() if v is not None}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extracted = OrderedDict()

    def add(self, expr, make, terms=None):
        """
        Without ``terms``: add ``expr`` to the mapper binding it to the symbol
        generated with the callback ``make``.
        With ``terms``: add the compound sub-expression made of ``terms`` to the
        mapper. ``terms`` is a list of one or more items in ``expr.args``.
        """
        if expr in self:
            return

        if not terms:
            self[expr] = self.extracted[expr] = make()
            return

        terms = as_list(terms)

        base = terms.pop(0)
        if terms:
            k = expr.func(base, *terms)
            try:
                symbol = self.extracted[k]
            except KeyError:
                symbol = self.extracted.setdefault(k, make())
            self[expr] = self.Uxsubmap.fromkeys(terms)
            self[expr][base] = symbol
        else:
            self[base] = self.extracted[base] = make()


def xreplace_indices(exprs, mapper, key=None):
    """
    Replace array indices in expressions.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        One or more expressions to which the replacement is applied.
    mapper : dict
        The substitution rules.
    key : list of symbols or callable
        An escape hatch to rule out some objects from the replacement.
        If a list, apply the replacement to the symbols in ``key`` only. If a
        callable, apply the replacement to a symbol S if and only if ``key(S)``
        gives True.
    """
    handle = flatten(retrieve_indexed(i) for i in as_tuple(exprs))
    if isinstance(key, Iterable):
        handle = [i for i in handle if i.base.label in key]
    elif callable(key):
        handle = [i for i in handle if key(i)]
    mapper = dict(zip(handle, [i.xreplace(mapper) for i in handle]))
    replaced = [uxreplace(i, mapper) for i in as_tuple(exprs)]
    return replaced if isinstance(exprs, Iterable) else replaced[0]


def pow_to_mul(expr):
    if expr.is_Atom or expr.is_Indexed:
        return expr
    elif expr.is_Pow:
        base, exp = expr.as_base_exp()
        if exp > 10 or exp < -10 or int(exp) != exp or exp == 0:
            # Large and non-integer powers remain untouched
            return expr
        elif exp == -1:
            # Reciprocals also remain untouched, but we traverse the base
            # looking for other Pows
            return expr.func(pow_to_mul(base), exp, evaluate=False)
        elif exp > 0:
            return sympy.Mul(*[base]*int(exp), evaluate=False)
        else:
            # SymPy represents 1/x as Pow(x,-1). Also, it represents
            # 2/x as Mul(2, Pow(x, -1)). So we shouldn't end up here,
            # but just in case SymPy changes its internal conventions...
            posexpr = sympy.Mul(*[base]*(-int(exp)), evaluate=False)
            return sympy.Pow(posexpr, -1, evaluate=False)
    else:
        return expr.func(*[pow_to_mul(i) for i in expr.args], evaluate=False)


def as_symbol(expr):
    """Cast to sympy.Symbol."""
    from devito.types import Dimension
    try:
        if expr.is_Symbol:
            return expr
    except AttributeError:
        pass
    try:
        return Number(expr)
    except (TypeError, ValueError):
        pass
    if isinstance(expr, str):
        return Symbol(expr)
    elif isinstance(expr, Dimension):
        return Symbol(expr.name)
    elif expr.is_Symbol:
        return expr
    elif isinstance(expr, Indexed):
        return expr.base.label
    else:
        raise TypeError("Cannot extract symbol from type %s" % type(expr))


AffineFunction = namedtuple("AffineFunction", "var, coeff, shift")


def split_affine(expr):
    """
    Split an affine scalar function into its three components, namely variable,
    coefficient, and translation from origin.

    Raises
    ------
    ValueError
        If ``expr`` is non affine.
    """
    if expr.is_Number:
        return AffineFunction(None, None, expr)

    # Handle super-quickly the calls like `split_affine(x+1)`, which are
    # the majority.
    if expr.is_Add and len(expr.args) == 2:
        if expr.args[0].is_Number and expr.args[1].is_Symbol:
            # SymPy deterministically orders arguments -- first numbers, then symbols
            return AffineFunction(expr.args[1], 1, expr.args[0])

    # Fallback
    poly = expr.as_poly()
    if not (poly.is_univariate and poly.is_linear) or not LM(poly).is_Symbol:
        raise ValueError
    return AffineFunction(LM(poly), LC(poly), poly.TC())


def indexify(expr):
    """
    Given a SymPy expression, return a new SymPy expression in which all
    AbstractFunction objects have been converted into Indexed objects.
    """
    mapper = {}
    for i in retrieve_functions(expr):
        try:
            if i.is_AbstractFunction:
                mapper[i] = i.indexify()
        except AttributeError:
            pass
    return expr.xreplace(mapper)


def aligned_indices(i, j, spacing):
    """
    Check if two indices are aligned. Two indices are aligned if they
    differ by an Integer*spacing.
    """
    try:
        return int((i - j)/spacing) == (i - j)/spacing
    except TypeError:
        return False


def subs_op_args(expr, args):
    """
    Substitute Operator argument values into an expression. `args` is
    expected to be as produced by `Operator.arguments` -- it can only
    contain string keys, with values that are not themselves expressions
    which will be substituted into.
    """
    return expr.subs({i.name: args[i.name] for i in expr.free_symbols if i.name in args})


def reuse_if_untouched(expr, args, evaluate=False):
    """
    Reconstruct `expr` iff any of the provided `args` is different than
    the corresponding arg in `expr.args`.
    """
    if all(a is b for a, b in zip(expr.args, args)):
        return expr
    else:
        return expr.func(*args, evaluate=evaluate)
