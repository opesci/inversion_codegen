from collections import Counter
from itertools import groupby, product

from devito.ir.clusters import Cluster, ClusterGroup, Queue
from devito.ir.support import TILABLE, Scope
from devito.passes.clusters.utils import cluster_pass
from devito.symbolics import pow_to_mul
from devito.tools import DAG, as_tuple, frozendict, timed_pass
from devito.types import Symbol

__all__ = ['Lift', 'fuse', 'optimize_pows', 'extract_increments']


class Lift(Queue):

    """
    Remove invariant Dimensions from Clusters to avoid redundant computation.

    Notes
    -----
    This is analogous to the compiler transformation known as
    "loop-invariant code motion".
    """

    @timed_pass(name='lift')
    def process(self, elements):
        return super(Lift, self).process(elements)

    def callback(self, clusters, prefix):
        if not prefix:
            # No iteration space to be lifted from
            return clusters

        hope_invariant = prefix[-1].dim._defines
        outer = set().union(*[i.dim._defines for i in prefix[:-1]])

        lifted = []
        processed = []
        for n, c in enumerate(clusters):
            # Increments prevent lifting
            if c.has_increments:
                processed.append(c)
                continue

            # Is `c` a real candidate -- is there at least one invariant Dimension?
            if any(d._defines & hope_invariant for d in c.used_dimensions):
                processed.append(c)
                continue

            impacted = set(processed) | set(clusters[n+1:])

            # None of the Functions appearing in a lifted Cluster can be written to
            if any(c.functions & set(i.scope.writes) for i in impacted):
                processed.append(c)
                continue

            # All of the inner Dimensions must appear in the write-to region
            # otherwise we would violate data dependencies. Consider
            #
            # 1)                 2)                          3)
            # for i              for i                       for i
            #   for x              for x                       for x
            #     r = f(a[x])        for y                       for y
            #                          r[x] = f(a[x, y])           r[x, y] = f(a[x, y])
            #
            # In 1) and 2) lifting is infeasible; in 3) the statement can be lifted
            # outside the `i` loop as `r`'s write-to region contains both `x` and `y`
            xed = {d._defines for d in c.used_dimensions if d not in outer}
            if not all(i & set(w.dimensions) for i, w in product(xed, c.scope.writes)):
                processed.append(c)
                continue

            # The contracted iteration and data spaces
            key = lambda d: d not in hope_invariant
            ispace = c.ispace.project(key).reset()
            dspace = c.dspace.project(key).reset()

            # Some properties are dropped
            properties = {d: v for d, v in c.properties.items() if key(d)}
            properties = {d: v - {TILABLE} for d, v in properties.items()}

            lifted.append(c.rebuild(ispace=ispace, dspace=dspace, properties=properties))

        return lifted + processed


class Fusion(Queue):

    """
    Fuse Clusters with compatible IterationSpace.
    """

    def __init__(self, toposort):
        super(Fusion, self).__init__()
        self.toposort = toposort

    def _make_key_hook(self, cgroup, level):
        assert level > 0
        assert len(cgroup.guards) == 1
        return (tuple(cgroup.guards[0].get(i.dim) for i in cgroup.itintervals[:level-1]),)

    def process(self, clusters):
        cgroups = [ClusterGroup(c, c.itintervals) for c in clusters]
        cgroups = self._process_fdta(cgroups, 1)
        clusters = ClusterGroup.concatenate(*cgroups)
        return clusters

    def callback(self, cgroups, prefix):
        # Toposort to maximize fusion
        if self.toposort:
            clusters = self._toposort(cgroups, prefix)
        else:
            clusters = ClusterGroup(cgroups)

        # Fusion
        processed = []
        for k, g in groupby(clusters, key=self._key):
            maybe_fusible = list(g)

            if len(maybe_fusible) == 1:
                processed.extend(maybe_fusible)
            else:
                try:
                    # Perform fusion
                    fused = Cluster.from_clusters(*maybe_fusible)
                    processed.append(fused)
                except ValueError:
                    # We end up here if, for example, some Clusters have same
                    # iteration Dimensions but different (partial) orderings
                    processed.extend(maybe_fusible)

        return [ClusterGroup(processed, prefix)]

    def _key(self, c):
        # Two Clusters/ClusterGroups are fusion candidates if their key is identical

        key = (frozenset(c.itintervals), c.guards)

        # We allow fusing Clusters/ClusterGroups with WaitLocks over different Locks,
        # while the WithLocks are to be kept separated (i.e. the remain separate tasks)
        if isinstance(c, Cluster):
            sync_locks = (c.sync_locks,)
        else:
            sync_locks = c.sync_locks
        for i in sync_locks:
            key += (frozendict({k: frozenset(type(i) if i.is_WaitLock else i for i in v)
                                for k, v in i.items()}),)

        return key

    def _toposort(self, cgroups, prefix):
        # Are there any ClusterGroups that could potentially be fused? If
        # not, do not waste time computing a new topological ordering
        counter = Counter(self._key(cg) for cg in cgroups)
        if not any(v > 1 for it, v in counter.most_common()):
            return ClusterGroup(cgroups)

        # Similarly, if all ClusterGroups have the same exact prefix and
        # use the same form of synchronization (if any at all), no need to
        # attempt a topological sorting
        if len(counter.most_common()) == 1:
            return ClusterGroup(cgroups)

        dag = self._build_dag(cgroups, prefix)

        def choose_element(queue, scheduled):
            # Heuristic: let `k0` be the key of the last scheduled node; then out of
            # the possible schedulable nodes we pick the one with key `k1` such that
            # `max_i : k0[:i] == k1[:i]` (i.e., the one with "the most similar key")
            if not scheduled:
                return queue.pop()
            key = self._key(scheduled[-1])
            for i in reversed(range(len(key) + 1)):
                candidates = [e for e in queue if self._key(e)[:i] == key[:i]]
                try:
                    # Ensure stability
                    e = min(candidates, key=lambda i: cgroups.index(i))
                except ValueError:
                    continue
                queue.remove(e)
                return e
            assert False

        return ClusterGroup(dag.topological_sort(choose_element))

    def _build_dag(self, cgroups, prefix):
        """
        A DAG representing the data dependences across the ClusterGroups within
        a given scope.
        """
        prefix = {i.dim for i in as_tuple(prefix)}

        dag = DAG(nodes=cgroups)
        for n, cg0 in enumerate(cgroups):
            for cg1 in cgroups[n+1:]:
                # A Scope to compute all cross-ClusterGroup anti-dependences
                rule = lambda i: i.is_cross
                scope = Scope(exprs=cg0.exprs + cg1.exprs, rules=rule)

                # Optimization: we exploit the following property:
                # no prefix => (edge <=> at least one (any) dependence)
                # to jump out of this potentially expensive loop as quickly as possible
                if not prefix and any(scope.d_all_gen()):
                    dag.add_edge(cg0, cg1)

                # Anti-dependences along `prefix` break the execution flow
                # (intuitively, "the loop nests are to be kept separated")
                # * All ClusterGroups between `cg0` and `cg1` must precede `cg1`
                # * All ClusterGroups after `cg1` cannot precede `cg1`
                elif any(i.cause & prefix for i in scope.d_anti_gen()):
                    for cg2 in cgroups[n:cgroups.index(cg1)]:
                        dag.add_edge(cg2, cg1)
                    for cg2 in cgroups[cgroups.index(cg1)+1:]:
                        dag.add_edge(cg1, cg2)
                    break

                # Any anti- and iaw-dependences impose that `cg1` follows `cg0`
                # while not being its immediate successor (unless it already is),
                # to avoid they are fused together (thus breaking the dependence)
                # TODO: the "not being its immediate successor" part *seems* to be
                # a work around to the fact that any two Clusters characterized
                # by anti-dependence should have been given a different stamp,
                # and same for guarded Clusters, but that is not the case (yet)
                elif any(scope.d_anti_gen()) or\
                        any(i.is_iaw for i in scope.d_output_gen()):
                    dag.add_edge(cg0, cg1)
                    index = cgroups.index(cg1) - 1
                    if index > n and self._key(cg0) == self._key(cg1):
                        dag.add_edge(cg0, cgroups[index])
                        dag.add_edge(cgroups[index], cg1)

                # Any flow-dependences along an inner Dimension (i.e., a Dimension
                # that doesn't appear in `prefix`) impose that `cg1` follows `cg0`
                elif any(not (i.cause and i.cause & prefix) for i in scope.d_flow_gen()):
                    dag.add_edge(cg0, cg1)

                # Clearly, output dependences must be honored
                elif any(scope.d_output_gen()):
                    dag.add_edge(cg0, cg1)

        return dag


@timed_pass()
def fuse(clusters, toposort=False):
    """
    Clusters fusion.

    If ``toposort=True``, then the Clusters are reordered to maximize the likelihood
    of fusion; the new ordering is computed such that all data dependencies are honored.
    """
    return Fusion(toposort=toposort).process(clusters)


@cluster_pass(mode='all')
def optimize_pows(cluster, *args):
    """
    Convert integer powers into Muls, such as ``a**2 => a*a``.
    """
    return cluster.rebuild(exprs=[pow_to_mul(e) for e in cluster.exprs])


@cluster_pass(mode='sparse')
def extract_increments(cluster, sregistry, *args):
    """
    Extract the RHSs of non-local tensor expressions performing an associative
    and commutative increment, and assign them to temporaries.
    """
    processed = []
    for e in cluster.exprs:
        if e.is_Increment and e.lhs.function.is_Input:
            handle = Symbol(name=sregistry.make_name(), dtype=e.dtype).indexify()
            if e.rhs.is_Number or e.rhs.is_Symbol:
                extracted = e.rhs
            else:
                extracted = e.rhs.func(*[i for i in e.rhs.args if i != e.lhs])
            processed.extend([e.func(handle, extracted, is_Increment=False),
                              e.func(e.lhs, handle)])
        else:
            processed.append(e)

    return cluster.rebuild(processed)
