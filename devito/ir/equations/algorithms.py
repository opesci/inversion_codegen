from collections.abc import Iterable
from operator import attrgetter

from devito.symbolics import (retrieve_functions, retrieve_indexed, split_affine,
                              uxreplace)
from devito.tools import PartialOrderTuple, filter_sorted, flatten, as_tuple
from devito.types import Dimension

__all__ = ['dimension_sort', 'lower_exprs']


def dimension_sort(expr):
    """
    Topologically sort the Dimensions in ``expr``, based on the order in which they
    appear within Indexeds.
    """

    def handle_indexed(indexed):
        relation = []
        for i in indexed.indices:
            try:
                maybe_dim = split_affine(i).var
                if isinstance(maybe_dim, Dimension):
                    relation.append(maybe_dim)
            except ValueError:
                # Maybe there are some nested Indexeds (e.g., the situation is A[B[i]])
                nested = flatten(handle_indexed(n) for n in retrieve_indexed(i))
                if nested:
                    relation.extend(nested)
                else:
                    # Fallback: Just insert all the Dimensions we find, regardless of
                    # what the user is attempting to do
                    relation.extend([d for d in filter_sorted(i.free_symbols)
                                     if isinstance(d, Dimension)])
        return tuple(relation)

    relations = {handle_indexed(i) for i in retrieve_indexed(expr)}

    # Add in any implicit dimension (typical of scalar temporaries, or Step)
    relations.add(expr.implicit_dims)

    # Add in leftover free dimensions (not an Indexed' index)
    extra = set([i for i in expr.free_symbols if isinstance(i, Dimension)])

    # Add in pure data dimensions (e.g., those accessed only via explicit values,
    # such as A[3])
    indexeds = retrieve_indexed(expr, deep=True)
    extra.update(set().union(*[set(i.function.dimensions) for i in indexeds]))

    # Enforce determinism
    extra = filter_sorted(extra, key=attrgetter('name'))

    # Add in implicit relations for parent dimensions
    # -----------------------------------------------
    # 1) Note that (d.parent, d) is what we want, while (d, d.parent) would be
    # wrong; for example, in `((t, time), (t, x, y), (x, y))`, `x` could now
    # preceed `time`, while `t`, and therefore `time`, *must* appear before `x`,
    # as indicated by the second relation
    implicit_relations = {(d.parent, d) for d in extra if d.is_Derived}
    # 2) To handle cases such as `((time, xi), (x,))`, where `xi` a SubDimension
    # of `x`, besides `(x, xi)`, we also have to add `(time, x)` so that we
    # obtain the desired ordering `(time, x, xi)`. W/o `(time, x)`, the ordering
    # `(x, time, xi)` might be returned instead, which would be non-sense
    implicit_relations.update({tuple(d.root for d in i) for i in relations})

    ordering = PartialOrderTuple(extra, relations=(relations | implicit_relations))

    return ordering


def lower_exprs(expressions, **kwargs):
    """
    Lowering an expression consists of the following passes:

        * Indexify functions;
        * Align Indexeds with the computational domain;
        * Apply user-provided substitution;

    Examples
    --------
    f(x - 2*h_x, y) -> f[xi + 2, yi + 4]  (assuming halo_size=4)
    """

    processed = []
    for expr in as_tuple(expressions):
        try:
            dimension_map = expr.subdomain.dimension_map
        except AttributeError:
            # Some Relationals may be pure SymPy objects, thus lacking the subdomain
            dimension_map = {}

        # Gather `Function`'s defined on a `SubDomain`
        fosd = set([f for f in retrieve_functions(expr, mode='unique')
                    if f.is_Function and f._subdomain])

        # Handle Functions (typical case)
        mapper = {**{f: f.indexify(lshift=True, subs=dimension_map)
                     for f in set(retrieve_functions(expr)).difference(fosd)},
                  **{f: f.indexify(lshift=True, subs=f._subdomain._access_map)
                     for f in fosd}}

        # Handle Indexeds (from index notation)
        for i in retrieve_indexed(expr):
            f = i.function

            # Introduce shifting to align with the computational domain
            indices = [(lower_exprs(a) + o) for a, o in
                       zip(i.indices, f._size_nodomain.left)]

            # Apply substitutions, if necessary
            if dimension_map:
                indices = [j.xreplace(dimension_map) for j in indices]

            mapper[i] = f.indexed[indices]

        subs = kwargs.get('subs')
        # Add dimensions map to the mapper in case dimensions are used
        # as an expression, i.e. Eq(u, x, subdomain=xleft)
        mapper.update(dimension_map)
        if subs:
            # Include the user-supplied substitutions, and use
            # `xreplace` for constant folding
            processed.append(expr.xreplace({**mapper, **subs}))
        else:
            processed.append(uxreplace(expr, mapper))

    if isinstance(expressions, Iterable):
        return processed
    else:
        assert len(processed) == 1
        return processed.pop()
