import cgen

from sympy import Min, Max
from devito.ir.iet import (Expression, List, Prodder, FindNodes, FindSymbols,
                           Transformer, filter_iterations, retrieve_iteration_tree)
from devito.ir.support import Forward
from devito.logger import warning
from devito.passes.iet.engine import iet_pass
from devito.symbolics import INT
from devito.tools import split, is_integer

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
    This pass is transforming an IET to iterate over "main" and "remainder" regions.
    IncrDimensions in the IET are not iterating over the whole domain, as they miss the
    the remainder regions. The remainder regions include a number of iterations
    that are less than a block's size iterations.
    This function rebuilds Iterations over IncrDimensions. Min/Max conditions
    are used to iterate over the whole domain using blocking.

    A simple example (blocking only), nested Iterations are
    transformed from:

    <Iteration x0_blk0; (x_m, x_M, x0_blk0_size)>
        <Iteration x; (x0_blk0, x0_blk0 + x0_blk0_size - 1, 1)>

    to:

    <Iteration x0_blk0; (x_m, x_M, x0_blk0_size)>
        <Iteration x; (x0_blk0, INT(Min(x_M, x0_blk0 + x0_blk0_size - 1)), 1)>

    """

    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.dim.is_Incr]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        outer, inner = split(iterations, lambda i: not i.dim.parent.is_Incr)

        # Get symbolic_max out of each outer dimension
        roots_max = {i.dim.root: i.symbolic_max for i in outer}

        # A dictionary to map maximum of parent dimensions
        # useful for hierarchical blocking
        proc_parents_max = {}

        # Process inner iterations
        for n, i in enumerate(inner):
            assert i.direction is Forward
            if (i.dim.parent in proc_parents_max and
               i.symbolic_size == i.dim.parent.step):
                # Special case: A parent dimension may have already been processed
                # as a member of 'inner' iterations. In this case we can use parent's
                # Max
                # Usually encountered in hierarchical blocking (BLOCKLEVELS > 1)
                it_max = proc_parents_max[i.dim.parent]
            else:
                # Most of the cases pass though this code:
                # Candidates for upper bound calculation are:
                # Candidate 1: symbolic_max of current iteration
                # e.g.
                # i.symbolic_max = x0_blk0 + x0_blk0_size
                symbolic_max = i.symbolic_max

                # Candidate 2: The domain max. Usualy it is the max of parent/root
                # dimension.
                # e.g. x_M
                # This may not always be true as the symbolic_size of an Iteration may
                # exceed the size of a parent's block size (e.g. after CIRE passes)
                # e.g.
                # i.dim.parent.step = x0_blk1_size
                # i.symbolic_size = x0_blk1_size + 4

                # For this case, proper margin should be allowed
                # in order not to drop iterations. So Candidate 2 is the maximum of the
                # root's max and the current iteration's required max
                # and instead of `x_M` we may need `x_M + 1` or `x_M + 2`
                root_max = roots_max[i.dim.root]

                lb_margin = i.symbolic_min - i.dim.symbolic_min
                size_margin = i.symbolic_size - i.dim.parent.step
                upper_margin = i.dim.parent.symbolic_max + size_margin + lb_margin

                # Domain max candidate
                # e.g. domain_max = Max(x_M + 1, x_M)
                domain_max = Max(upper_margin, root_max)

                # Finally our upper bound is the minimum of upper bound candidates
                # e.g. upper_bound = Min(x0_blk0 + x0_blk0_size, domain_max)
                it_max = Min(symbolic_max, domain_max)

            # Store the selected maximum of this iteration's dimension for
            # possible reference in case of children iterations
            # Usually encountered in subdims and hierarchical blocking
            proc_parents_max[i.dim] = it_max

            mapper[i] = i._rebuild(limits=(i.symbolic_min, INT(it_max), i.step))

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {'efuncs': []}


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
