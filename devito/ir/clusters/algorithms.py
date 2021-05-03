from functools import partial
from itertools import groupby

import numpy as np
import sympy

from devito.exceptions import InvalidOperator
from devito.ir.support import Any, Backward, Forward, IterationSpace
from devito.ir.clusters.analysis import analyze
from devito.ir.clusters.cluster import Cluster, ClusterGroup, Guard
from devito.ir.clusters.queue import Queue, QueueStateful
from devito.parameters import configuration
from devito.symbolics import uxreplace, xreplace_indices
from devito.tools import (DefaultOrderedDict, as_mapper, flatten, frozendict,
                          is_integer, timed_pass)
from devito.types import ModuloDimension

__all__ = ['clusterize']


def clusterize(exprs):
    """
    Turn a sequence of LoweredEqs into a sequence of Clusters.
    """
    # Initialization
    clusters = [Cluster(e, e.ispace, e.dspace) for e in exprs]

    # Setup the IterationSpaces based on data dependence analysis
    clusters = Schedule().process(clusters)

    # Handle SteppingDimensions
    clusters = Stepper().process(clusters)

    # Handle ConditionalDimensions
    clusters = guard(clusters)

    # Determine relevant computational properties (e.g., parallelism)
    clusters = analyze(clusters)

    return ClusterGroup(clusters)


class Schedule(QueueStateful):

    """
    This special Queue produces a new sequence of "scheduled" Clusters, which
    means that:

        * The iteration direction along each Dimension of each Cluster is such
          that the information "naturally flows from one iteration to another".
          For example, in `u[t+1, x] = u[t, x]`, the iteration Dimension `t`
          gets assigned the `Forward` direction, to honor the flow-dependence
          along `t`. Instead, in `u[t-1, x] = u[t, x]`, `t` gets assigned the
          `Backward` direction. This simple rule ensures that when we evaluate
          the LHS, the information on the RHS is up-to-date.

        * If a Cluster has both a flow- and an anti-dependence along a given
          Dimension `x`, then `x` is assigned the `Forward` direction but its
          IterationSpace is _lifted_ such that it cannot be fused with any
          other Clusters within the same iteration Dimension `x`. For example,
          consider the following coupled statements:

            - `u[t+1, x] = f(u[t, x])`
            - `v[t+1, x] = g(v[t, x], u[t, x], u[t+1, x], u[t+2, x])`

          The first statement has a flow-dependence along `t`, while the second
          one has both a flow- and an anti-dependence along `t`, hence the two
          statements will ultimately be kept in separate Clusters and then
          scheduled to different loop nests.

        * If *all* dependences across two Clusters along a given Dimension are
          backward carried depedences, then the IterationSpaces are _lifted_
          such that the two Clusters cannot be fused. This is to maximize
          the number of parallel Dimensions. Essentially, this is what low-level
          compilers call "loop fission" -- only that here it occurs at a much
          higher level of abstraction. For example:

            - `u[x+1] = w[x] + v[x]`
            - `v[x] = u[x] + w[x]

          Here, the two statements will ultimately be kept in separate Clusters
          and then scheduled to different loops; this way, `x` will be a parallel
          Dimension in both Clusters.
    """

    @timed_pass(name='schedule')
    def process(self, clusters):
        return self._process_fdta(clusters, 1)

    def callback(self, clusters, prefix, backlog=None, known_break=None):
        if not prefix:
            return clusters

        known_break = known_break or set()
        backlog = backlog or []

        # Take the innermost Dimension -- no other Clusters other than those in
        # `clusters` are supposed to share it
        candidates = prefix[-1].dim._defines

        scope = self._fetch_scope(clusters)

        # Handle the nastiest case -- ambiguity due to the presence of both a
        # flow- and an anti-dependence.
        #
        # Note: in most cases, `scope.d_anti.cause == {}` -- either because
        # `scope.d_anti == {}` or because the few anti dependences are not carried
        # in any Dimension. We exploit this observation so that we only compute
        # `d_flow`, which instead may be expensive, when strictly necessary
        maybe_break = scope.d_anti.cause & candidates
        if len(clusters) > 1 and maybe_break:
            require_break = scope.d_flow.cause & maybe_break
            if require_break:
                backlog = [clusters[-1]] + backlog
                # Try with increasingly smaller ClusterGroups until the ambiguity is gone
                return self.callback(clusters[:-1], prefix, backlog, require_break)

        # Schedule Clusters over different IterationSpaces if this increases parallelism
        if configuration["loop-splitting"]:
            for i in range(1, len(clusters)):
                if self._break_for_parallelism(scope, candidates, i):
                    return self.callback(clusters[:i], prefix, clusters[i:] + backlog,
                                         candidates | known_break)

        # Compute iteration direction
        idir = {d: Backward for d in candidates if d.root in scope.d_anti.cause}
        if maybe_break:
            idir.update({d: Forward for d in candidates if d.root in scope.d_flow.cause})
        idir.update({d: Forward for d in candidates if d not in idir})

        # Enforce iteration direction on each Cluster
        processed = []
        for c in clusters:
            ispace = IterationSpace(c.ispace.intervals, c.ispace.sub_iterators,
                                    {**c.ispace.directions, **idir})
            processed.append(c.rebuild(ispace=ispace))

        if not backlog:
            return processed

        # Handle the backlog -- the Clusters characterized by flow- and anti-dependences
        # along one or more Dimensions
        idir = {d: Any for d in known_break}
        for i, c in enumerate(list(backlog)):
            ispace = IterationSpace(c.ispace.intervals.lift(known_break),
                                    c.ispace.sub_iterators,
                                    {**c.ispace.directions, **idir})
            dspace = c.dspace.lift(known_break)
            backlog[i] = c.rebuild(ispace=ispace, dspace=dspace)

        return processed + self.callback(backlog, prefix)

    def _break_for_parallelism(self, scope, candidates, i):
        # `test` will be True if there's at least one data-dependence that would
        # break parallelism
        test = False
        for d in scope.d_from_access_gen(scope.a_query(i)):
            if d.is_local or d.is_storage_related(candidates):
                # Would break a dependence on storage
                return False
            if any(d.is_carried(i) for i in candidates):
                if (d.is_flow and d.is_lex_negative) or (d.is_anti and d.is_lex_positive):
                    # Would break a data dependence
                    return False
            test = test or (bool(d.cause & candidates) and not d.is_lex_equal)
        return test


@timed_pass()
def guard(clusters):
    """
    Split Clusters containing conditional expressions into separate Clusters.
    """
    processed = []
    for c in clusters:
        # Group together consecutive expressions with same ConditionalDimensions
        for cds, g in groupby(c.exprs, key=lambda e: tuple(e.conditionals)):
            exprs = list(g)

            if not cds:
                processed.append(c.rebuild(exprs=exprs))
                continue

            # Chain together all conditions from all expressions in `c`
            guards = {}
            for cd in cds:
                condition = []
                for e in exprs:
                    try:
                        condition.append(e.conditionals[cd])
                        break
                    except KeyError:
                        pass
                condition = sympy.And(*condition, evaluate=False)
                guards[cd.parent] = Guard(condition, cd.brk)

            # Construct a guarded Cluster
            processed.append(c.rebuild(exprs=exprs, guards=guards))

    return ClusterGroup(processed)


class Stepper(Queue):

    """
    Produce a new sequence of Clusters in which the IterationSpaces carry the
    sub-iterators induced by a SteppingDimension.
    """

    def process(self, clusters):
        return self._process_fdta(clusters, 1)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        subiters = flatten([c.ispace.sub_iterators.get(d, []) for c in clusters])
        subiters = {i for i in subiters if i.is_Stepping}
        if not subiters:
            return clusters

        # Collect the index access functions along `d`, e.g., `t + 1` where `t` is
        # a SteppingDimension for `d = time`
        mapper = DefaultOrderedDict(lambda: DefaultOrderedDict(set))
        for c in clusters:
            indexeds = [a.indexed for a in c.scope.accesses if a.function.is_Tensor]

            for i in indexeds:
                try:
                    iaf = i.indices[d]
                except KeyError:
                    continue

                # Sanity checks
                sis = iaf.free_symbols & subiters
                if len(sis) == 0:
                    continue
                elif len(sis) == 1:
                    si = sis.pop()
                else:
                    raise InvalidOperator("Cannot use multiple SteppingDimensions "
                                          "to index into a Function")
                size = i.function.shape_allocated[d]
                assert is_integer(size)

                mapper[size][si].add(iaf)

        # Construct the ModuloDimensions
        mds = []
        for size, v in mapper.items():
            for si, iafs in list(v.items()):
                # Offsets are sorted so that the semantic order (t0, t1, t2) follows
                # SymPy's index ordering (t, t-1, t+1) afer modulo replacement so
                # that associativity errors are consistent. This corresponds to
                # sorting offsets {-1, 0, 1} as {0, -1, 1} assigning -inf to 0
                siafs = sorted(iafs, key=lambda i: -np.inf if i - si == 0 else (i - si))

                for iaf in siafs:
                    name = '%s%d' % (si.name, len(mds))
                    offset = uxreplace(iaf, {si: d.root})
                    mds.append(ModuloDimension(name, si, offset, size, origin=iaf))

        # Replacement rule for ModuloDimensions
        def rule(size, e):
            try:
                return e.function.shape_allocated[d] == size
            except (AttributeError, KeyError):
                return False

        # Reconstruct the Clusters
        processed = []
        for c in clusters:
            # Apply substitutions to expressions
            # Note: In an expression, there could be `u[t+1, ...]` and `v[t+1,
            # ...]`, where `u` and `v` are TimeFunction with circular time
            # buffers (save=None) *but* different modulo extent. The `t+1`
            # indices above are therefore conceptually different, so they will
            # be replaced with the proper ModuloDimension through two different
            # calls to `xreplace_indices`
            exprs = c.exprs
            groups = as_mapper(mds, lambda d: d.modulo)
            for size, v in groups.items():
                mapper = {md.origin: md for md in v}

                func = partial(xreplace_indices, mapper=mapper, key=partial(rule, size))
                exprs = [e.apply(func) for e in exprs]

            # Augment IterationSpace
            ispace = IterationSpace(c.ispace.intervals,
                                    {**c.ispace.sub_iterators, **{d: tuple(mds)}},
                                    c.ispace.directions)

            processed.append(c.rebuild(exprs=exprs, ispace=ispace))

        return processed
