import cgen

from devito.ir.iet import (Expression, List, Prodder, FindNodes, FindSymbols,
                           Transformer, filter_iterations,
                           retrieve_iteration_tree)
from devito.passes.iet.engine import iet_pass
from devito.symbolics import INT
from devito.tools import split
from devito.logger import warning
from sympy import Min, Max

__all__ = ['avoid_denormals', 'hoist_prodders', 'relax_incr_dimensions', 'is_on_device']


@iet_pass
def avoid_denormals(iet):
    """
    Introduce nodes in the Iteration/Expression tree that will expand to C
    macros telling the CPU to flush denormal numbers in hardware. Denormals
    are normally flushed when using SSE-based instruction sets, except when
    compiling shared objects.
    """
    if iet.is_ElementalFunction:
        return iet, {}

    header = (cgen.Comment('Flush denormal numbers to zero in hardware'),
              cgen.Statement('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)'),
              cgen.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)'),
              cgen.Line())
    iet = iet._rebuild(body=(List(header=header),) + iet.body)
    return iet, {'includes': ('xmmintrin.h', 'pmmintrin.h')}


@iet_pass
def hoist_prodders(iet):
    """
    Move Prodders within the outer levels of an Iteration tree.
    """
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        for prodder in FindNodes(Prodder).visit(tree.root):
            if prodder._periodic:
                try:
                    key = lambda i: i.dim.is_Incr and i.dim.step != 1
                    candidate = filter_iterations(tree, key)[-1]
                except IndexError:
                    # Fallback: use the outermost Iteration
                    candidate = tree.root
                mapper[candidate] = candidate._rebuild(nodes=(candidate.nodes +
                                                              (prodder._rebuild(),)))
                mapper[prodder] = None

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}


@iet_pass
def relax_incr_dimensions(iet, **kwargs):
    """
    Recast Iterations over IncrDimensions as ElementalFunctions; insert
    ElementalCalls to iterate over the "main" and "remainder" regions induced
    by the IncrDimensions.
    """
    efuncs = []
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.dim.is_Incr]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        # Split iterations to outer and inner
        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Incr)

        # Compute the iteration ranges
        ranges = []
        for i in outer:
            maxb = i.symbolic_max - (i.symbolic_size % i.dim.step)
            ranges.append(((i.symbolic_min, maxb, i.dim.step),
                           (maxb + 1, i.symbolic_max, i.symbolic_max - maxb)))

        # Remove any offsets
        # E.g., `x = x_m + 2 to x_M - 2` --> `x = x_m to x_M`
        outer = [i._rebuild(limits=(i.dim.root.symbolic_min, i.dim.root.symbolic_max,
                                    i.step))
                 for i in outer]

        new_iters = []
        new_iters = [i for i in outer]
        nodes1 = []
        levels_max = {}

        for n, i in enumerate(inner):

            b0a = (i.dim.parent.symbolic_max - i.dim.parent.step -
                   i.dim.symbolic_min + i.symbolic_size + i.symbolic_min)

            b0b = i.dim.parent.symbolic_max

            try:
                rangemax = ranges[n][-1][1]
            except:
                rangemax = b0b

            b1 = Max(b0a, rangemax)

            if i.dim.parent in levels_max.keys() and i.symbolic_size <= i.dim.parent.step:
                b2 = levels_max[i.dim.parent]
                ub = Min(b2)
            else:
                ub = Min(i.symbolic_max, b1)

            levels_max[i.dim] = ub

            new_inner = i._rebuild(limits=(i.symbolic_min, INT(ub), i.step))
            nodes1.append(new_inner)
            new_iters.append(i)
            mapper[i] = new_inner
            inner[n] = i._rebuild(limits=(i.symbolic_min, ub, i.step))

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {'efuncs': efuncs}


def is_on_device(maybe_symbol, gpu_fit, only_writes=False):
    """
    True if all given Functions are allocated in the device memory, False otherwise.

    Parameters
    ----------
    maybe_symbol : Indexed or Function or Node
        The inspected object. May be a single Indexed or Function, or even an
        entire piece of IET.
    gpu_fit : list of Function
        The Function's which are known to definitely fit in the device memory. This
        information is given directly by the user through the compiler option
        `gpu-fit` and is propagated down here through the various stages of lowering.
    only_writes : bool, optional
        Only makes sense if `maybe_symbol` is an IET. If True, ignore all Function's
        that do not appear on the LHS of at least one Expression. Defaults to False.
    """
    try:
        functions = (maybe_symbol.function,)
    except AttributeError:
        assert maybe_symbol.is_Node
        iet = maybe_symbol
        functions = set(FindSymbols().visit(iet))
        if only_writes:
            expressions = FindNodes(Expression).visit(iet)
            functions &= {i.write for i in expressions}

    fsave = [f for f in functions if f.is_TimeFunction and is_integer(f.save)]
    if 'all-fallback' in gpu_fit and fsave:
        warning("TimeFunction %s assumed to fit the GPU memory" % fsave)
        return True

    return all(f in gpu_fit for f in fsave)
