import cgen

from sympy import Min, Max

from devito.ir.iet import (Expression, List, Prodder, FindNodes, FindSymbols,
                           Transformer, filter_iterations, retrieve_iteration_tree)
from devito.logger import warning
from devito.passes.iet.engine import iet_pass
from devito.symbolics import INT
from devito.tools import split

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
    Recast Iteration bounds using min/max conditions to iterate over the
    domain. This function sets the limits of a tree's Iterations.
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

        # Get symbolic_max out of each outer dimension
        ranges = {}
        for i in outer:
            ranges[i.dim.root] = i.symbolic_max

        # A dictionary to map maximum of parent dimensions
        # useful for hierarchical blocking
        proc_parents_max = {}

        # Process inner iterations
        for n, i in enumerate(inner):

            # Candidate 1: Retrieve parent's max from outer limits
            parent_max = ranges[i.dim.root]

            # Candidate 2: The symbolic_size of an inner Iteration
            # may exceed the size of a parent's
            # block size. Proper margin should be allowed
            # For cases where upperbound - lowerbound > block_size:
            # i.symbolic_size > i.dim.parent.step
            low_margin = i.symbolic_min - i.dim.symbolic_min
            size_extent = i.symbolic_size - i.dim.parent.step

            upper_ext = i.dim.parent.symbolic_max + size_extent + low_margin

            # Maximum of upper bound candidates
            it_max = Max(upper_ext, parent_max)

            # In case of hierarchical blocking we should take care not to exceed the
            # maximum of the parent dimension.
            # In case parent dim has been processed in the current tree:
            # if iteration size < parent block_size/step
            #    upper bound is the parent_max
            # else/otherwise
            #    upper bound is the minimum of it_max and parent_max

            if (i.dim.parent in proc_parents_max.keys() and
               i.symbolic_size <= i.dim.parent.step):
                upper_bound = Min(proc_parents_max[i.dim.parent])
            else:
                upper_bound = Min(i.symbolic_max, it_max)

            # Store selected maximum of this iteration's dimension for
            # children iteration reference in hierarchical blocking
            proc_parents_max[i.dim] = upper_bound

            mapper[i] = i._rebuild(limits=(i.symbolic_min, INT(upper_bound), i.step))

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
