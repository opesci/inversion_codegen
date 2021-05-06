from collections import Counter

from devito.logger import warning
from devito.symbolics.queries import q_routine
from devito.symbolics.search import retrieve_xops, search
from devito.tools import as_tuple, flatten

__all__ = ['compare_ops', 'count', 'estimate_cost']


def compare_ops(e1, e2):
    """
    True if the two input expressions perform the same arithmetic operations
    over the same input "operands", False otherwise.

    An operand here is anything that can appear as a leaf in a SymPy
    expression, but in the case of an Indexed only the labels are compared,
    while the indices are ignored.

    Examples
    --------
    >>> from devito import Dimension, Grid, Function
    >>> grid = Grid(shape=(4,))
    >>> x = grid.dimensions[0]
    >>> y = Dimension(name='y')
    >>> u = Function(name='u', grid=grid)
    >>> v = Function(name='v', grid=grid)
    >>> compare_ops(u[x] + u[x+1], u[x] + u[x-1])
    True
    >>> compare_ops(u[x] + u[x+1], u[x] - u[x+1])
    False
    >>> compare_ops(u[x] + u[x+1], u[x] * u[x+1])
    False
    >>> compare_ops(u[x] + u[x+1], u[x] + v[x+1])
    False
    >>> compare_ops(u[x] + u[x+1], u[x] + u[y+10])
    True
    """
    if type(e1) == type(e2) and len(e1.args) == len(e2.args):
        if e1.is_Atom:
            return True if e1 == e2 else False
        elif e1.is_Indexed and e2.is_Indexed:
            return True if e1.base == e2.base else False
        else:
            for a1, a2 in zip(e1.args, e2.args):
                if not compare_ops(a1, a2):
                    return False
            return True
    else:
        return False


def count(exprs, query):
    """
    Return a mapper ``{(k, v)}`` where ``k`` is a sub-expression in ``exprs``
    matching ``query`` and ``v`` is the number of its occurrences.
    """
    mapper = Counter()
    for expr in as_tuple(exprs):
        mapper.update(Counter(search(expr, query, 'all', 'bfs')))
    return dict(mapper)


def estimate_cost(exprs, estimate=False):
    """
    Estimate the operation count of an expression.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        One or more expressions for which the operation count is calculated.
    estimate : bool, optional
        Defaults to False; if True, the following rules are applied in order:
            * An operation involving only `Constant`'s counts as 1 ops.
            * Trascendental functions (e.g., cos, sin, ...) count as 50 ops.
            * Divisions (powers with a negative exponent) count as 25 ops.
            * Powers with integer exponent `n>0` count as n-1 ops (as if
              it were a chain of multiplications).
    """
    elementary_cost = 100
    pow_cost = 50
    div_cost = 5

    try:
        # Is it a plain symbol/array ?
        if exprs.is_Atom or exprs.is_Indexed or exprs.is_AbstractFunction:
            return 0
    except AttributeError:
        pass
    try:
        # Is it a dict ?
        exprs = exprs.values()
    except AttributeError:
        try:
            # Could still be a list of dicts
            exprs = flatten([i.values() for i in exprs])
        except (AttributeError, TypeError):
            pass
    try:
        # At this point it must be a list of SymPy objects
        # We don't use SymPy's count_ops because we do not count integer arithmetic
        # (e.g., array index functions such as i+1 in A[i+1])
        # Also, the routine below is *much* faster than count_ops
        exprs = [i.rhs if i.is_Equality else i for i in as_tuple(exprs)]
        operations = flatten(retrieve_xops(i) for i in exprs)
        flops = 0
        for op in operations:
            if op.is_Function:
                if estimate and q_routine(op):
                    flops += elementary_cost
                else:
                    flops += 1
            elif op.is_Pow:
                if estimate:
                    if op.exp.is_Number:
                        if op.exp < 0:
                            flops += div_cost
                        elif op.exp == 0 or op.exp == 1:
                            flops += 0
                        elif op.exp.is_Integer:
                            # Natural pows a**b are estimated as b-1 Muls
                            flops += int(op.exp) - 1
                        else:
                            flops += pow_cost
                    else:
                        flops += pow_cost
                else:
                    flops += 1
            else:
                flops += len(op.args) - (1 + sum(True for i in op.args if i.is_Integer))
        return flops
    except:
        warning("Cannot estimate cost of `%s`" % str(exprs))
        return 0
