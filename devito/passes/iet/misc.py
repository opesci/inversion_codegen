from itertools import product

import cgen

from devito.ir.iet import (Expression, List, Prodder, FindNodes, FindSymbols,
                           Transformer, make_efunc, compose_nodes, filter_iterations,
                           retrieve_iteration_tree, IsPerfectIteration)
from devito.passes.iet.engine import iet_pass
from devito.tools import flatten, is_integer, split
from devito.logger import warning

__all__ = ['avoid_denormals', 'hoist_prodders', 'relax_incr_dimensions',
           'elementify', 'is_on_device', 'restrict_with_cpp']


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
def restrict_with_cpp(iet):
    if iet.is_ElementalFunction:
        return iet, {}

    return iet, {'headers': [('restrict', '__restrict')]}


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
    sregistry = kwargs['sregistry']

    efuncs = []
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.dim.is_Incr]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

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

        # Create the ElementalFunction
        name = sregistry.make_name(prefix="bf")
        body = compose_nodes(outer)
        dynamic_parameters = flatten((i.symbolic_bounds, i.step) for i in outer)
        dynamic_parameters.extend([i.step for i in inner if not is_integer(i.step)])
        efunc = make_efunc(name, body, dynamic_parameters)

        efuncs.append(efunc)

        # Create the ElementalCalls
        calls = []
        for p in product(*ranges):
            dynamic_args_mapper = {}
            for i, (m, M, b) in zip(outer, p):
                dynamic_args_mapper[i.symbolic_min] = m
                dynamic_args_mapper[i.symbolic_max] = M
                dynamic_args_mapper[i.step] = b
                for j in inner:
                    if j.dim.root is i.dim.root and not is_integer(j.step):
                        value = j.step if b is i.step else b
                        dynamic_args_mapper[j.step] = (value,)
            calls.append(efunc.make_call(dynamic_args_mapper))

        mapper[root] = List(body=calls)

    iet = Transformer(mapper).visit(iet)

    return iet, {'efuncs': efuncs}


@iet_pass
def elementify(iet, **kwargs):
    sregistry = kwargs['sregistry']

    efuncs = []
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        iterations = [i for i in tree if i.is_ParallelRelaxed]
        if not iterations:
            continue

        root = iterations[0]
        if root in mapper:
            continue

        #TODO: This is work in progress...
        assert IsPerfectIteration().visit(root)

        name = sregistry.make_name(prefix="kernel")
        efunc = make_efunc(name, root)

        mapper[root] = efunc.make_call()
        efuncs.append(efunc)

    iet = Transformer(mapper).visit(iet)

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
