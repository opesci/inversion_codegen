from devito.ir.iet import Iteration, List, IterationTree, FindSections, FindSymbols
from devito.symbolics import Literal, Macro
from devito.tools import flatten, split
from devito.types import Array, LocalObject

__all__ = ['filter_iterations', 'retrieve_iteration_tree', 'compose_nodes',
           'derive_parameters', 'diff_parameters']


def retrieve_iteration_tree(node, mode='normal'):
    """
    A list with all Iteration sub-trees within an IET.

    Examples
    --------
    Given the Iteration tree:

        .. code-block:: c

           Iteration i
             expr0
             Iteration j
               Iteration k
                 expr1
             Iteration p
               expr2

    Return the list: ::

        [(Iteration i, Iteration j, Iteration k), (Iteration i, Iteration p)]

    Parameters
    ----------
    iet : Node
        The searched Iteration/Expression tree.
    mode : str, optional
        - ``normal``
        - ``superset``: Iteration trees that are subset of larger iteration trees
                        are dropped.
    """
    assert mode in ('normal', 'superset')

    trees = [IterationTree(i) for i in FindSections().visit(node) if i]
    if mode == 'normal':
        return trees
    else:
        match = []
        for i in trees:
            if any(set(i).issubset(set(j)) for j in trees if i != j):
                continue
            match.append(i)
        return IterationTree(match)


def filter_iterations(tree, key=lambda i: i):
    """
    Return the first sub-sequence of consecutive Iterations such that
    ``key(iteration)`` is True.
    """
    filtered = []
    for i in tree:
        if key(i):
            filtered.append(i)
        elif len(filtered) > 0:
            break
    return filtered


def compose_nodes(nodes, retrieve=False):
    """
    Build an IET by nesting ``nodes``.
    
    Parameters
    ----------
    nodes : A list of nodes
        Input nodes to be nested.
    retrieve : 
    
    """
    l = list(nodes)
    tree = []

    if not isinstance(l[0], Iteration):
        # Nothing to compose
        body = flatten(l)
        body = List(body=body) if len(body) > 1 else body[0]
    else:
        body = l.pop(-1)
        while l:
            handle = l.pop(-1)
            body = handle._rebuild(body, **handle.args_frozen)
            tree.append(body)

    if retrieve is True:
        tree = list(reversed(tree))
        return body, tree
    else:
        return body


def derive_parameters(iet, drop_locals=False):
    """
    Derive all input parameters (function call arguments) from an IET
    by collecting all symbols not defined in the tree itself.
    """
    # Pick all free symbols and symbolic functions from the kernel
    functions = FindSymbols('symbolics').visit(iet)
    free_symbols = FindSymbols('free-symbols').visit(iet)

    # Filter out function base symbols and use real function objects
    function_names = set(flatten([(s.name, s._C_name) for s in functions]))
    symbols = [s for s in free_symbols if s.name not in function_names]
    symbols = functions + symbols

    defines = [s.name for s in FindSymbols('defines').visit(iet)]
    parameters = tuple(s for s in symbols if s.name not in defines)

    # Drop globally-visible objects
    parameters = [p for p in parameters if not isinstance(p, (Literal, Macro))]

    # Maybe filter out all other compiler-generated objects
    if drop_locals:
        parameters = [p for p in parameters if not isinstance(p, (Array, LocalObject))]

    return parameters


def diff_parameters(iet, root):
    """
    Derive the parameters of a sub-IET, `iet`, within a Callable, `root`, and
    split them into two groups:

        * the "read-only" parameters, and
        * the "dynamic" parameters, whose value changes at some point in `root`.
    """
    # TODO: this is currently very rudimentary
    required = derive_parameters(iet)

    known = set(root.parameters) | set(i for i in required if i.is_Array)

    parameters, dynamic_parameters = split(required, lambda i: i in known)

    return required, parameters, dynamic_parameters
