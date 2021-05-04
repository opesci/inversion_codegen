from collections import Counter, OrderedDict, defaultdict, namedtuple
from functools import singledispatch
from itertools import groupby

from cached_property import cached_property
import numpy as np
import sympy

from devito.finite_differences import EvalDerivative
from devito.ir import (SEQUENTIAL, PARALLEL_IF_PVT, ROUNDABLE, DataSpace,
                       Forward, IterationInstance, IterationSpace, Interval,
                       Cluster, Queue, IntervalGroup, LabeledVector,
                       detect_accesses, build_intervals, normalize_properties,
                       relax_properties)
from devito.passes.clusters.utils import timed_pass
from devito.symbolics import (Uxmapper, compare_ops, estimate_cost, q_constant,
                              reuse_if_untouched, retrieve_indexed, search, uxreplace)
from devito.tools import as_mapper, as_tuple, flatten, frozendict, generator, split
from devito.types import (Array, TempFunction, Eq, Symbol, ModuloDimension,
                          CustomDimension, IncrDimension, Indexed)

__all__ = ['cire']


@timed_pass(name='cire')
def cire(clusters, mode, sregistry, options, platform):
    """
    Cross-iteration redundancies elimination.

    Parameters
    ----------
    cluster : list of Cluster
        Input Clusters, subject of the optimization pass.
    mode : str
        The transformation mode. Accepted: ['invariants', 'sops'].
        * 'invariants' is for sub-expressions that are invariant w.r.t. one or
          more Dimensions.
        * 'sops' stands for sums-of-products, that is redundancies are searched
          across all expressions in sum-of-product form.
    sregistry : SymbolRegistry
        The symbol registry, to create unique temporary names.
    options : dict
        The optimization options.
        Accepted: ['min-storage', 'cire-maxpar', 'cire-rotate'].
        * 'min-storage': if True, the pass will try to minimize the amount of
          storage introduced for the tensor temporaries. This might also reduce
          the operation count. On the other hand, this might affect fusion and
          therefore data locality. Defaults to False (legacy).
        * 'cire-maxpar': if True, privilege parallelism over working set size,
          that is the pass will try to create as many parallel loops as possible,
          even though this will require more space (Dimensions) for the temporaries.
          Defaults to False.
        * 'cire-rotate': if True, the pass will use modulo indexing for the
          outermost Dimension iterated over by the temporaries. This will sacrifice
          a parallel loop for a reduced working set size. Defaults to False (legacy).
    platform : Platform
        The underlying platform. Used to optimize the shape of the introduced
        tensor symbols.

    Examples
    --------
    1) 'invariants'. Here's an expensive expression invariant w.r.t. `t`

    t0 = (cos(a[x,y,z])*sin(b[x,y,z]))*c[t,x,y,z]

    which after CIRE becomes

    t1[x,y,z] = cos(a[x,y,z])*sin(b[x,y,z])
    t0 = t1[x,y,z]*c[t,x,y,z]

    2) 'sops'. Below we see two expressions in sum-of-product form (in this
    case, the sum degenerates to a single product).

    t0 = 2.0*a[x,y,z]*b[x,y,z]
    t1 = 3.0*a[x,y,z+1]*b[x,y,z+1]

    CIRE detects that these two expressions are actually redundant and rewrites
    them as:

    t2[x,y,z] = a[x,y,z]*b[x,y,z]
    t0 = 2.0*t2[x,y,z]
    t1 = 3.0*t2[x,y,z+1]
    """
    modes = {
        CireInvariants.optname: CireInvariants,
        CireSops.optname: CireSops,
    }

    return modes[mode](sregistry, options, platform).process(clusters)


class CireTransformer(object):

    """
    Abstract base class for transformers implementing a CIRE variant.
    """

    optname = None

    def __init__(self, sregistry, options, platform):
        self.sregistry = sregistry
        self.platform = platform

        self.opt_minstorage = options['min-storage']
        self.opt_rotate = options['cire-rotate']
        self.opt_ftemps = options['cire-ftemps']
        self.opt_mingain = options['cire-mingain']

    def _aliases_from_clusters(self, clusters, exclude, meta):
        exprs = flatten([c.exprs for c in clusters])

        # [Clusters]_n -> [Schedule]_m
        variants = []
        for mapper in self._generate(exprs, exclude):
            # Clusters -> AliasList
            found = collect(mapper.extracted, meta.ispace, self.opt_minstorage)
            pexprs, aliases = choose(found, exprs, mapper, self.opt_mingain)
            if not aliases:
                continue

            # AliasList -> Schedule
            schedule = lower_aliases(aliases, meta, self.opt_maxpar)

            variants.append(Variant(schedule, pexprs))

        # [Schedule]_m -> Schedule (s.t. best memory/flops trade-off)
        if not variants:
            return []
        schedule, exprs = pick_best(variants, self.opt_schedule_strategy, self._betterv)

        # Schedule -> Schedule (optimization)
        if self.opt_rotate:
            schedule = optimize_schedule_rotations(schedule, self.sregistry)
        schedule = optimize_schedule_padding(schedule, meta, self.platform)

        # Schedule -> [Clusters]_k
        processed, subs = lower_schedule(schedule, meta, self.sregistry, self.opt_ftemps)

        # [Clusters]_k -> [Clusters]_{k+n}
        for c in clusters:
            n = len(c.exprs)
            cexprs, exprs = exprs[:n], exprs[n:]

            cexprs = [uxreplace(e, subs) for e in cexprs]

            ispace = c.ispace.augment(schedule.dmapper)
            ispace = ispace.augment(schedule.rmapper)

            accesses = detect_accesses(cexprs)
            parts = {k: IntervalGroup(build_intervals(v)).relaxed
                     for k, v in accesses.items() if k}
            dspace = DataSpace(c.dspace.intervals, parts)

            processed.append(c.rebuild(exprs=cexprs, ispace=ispace, dspace=dspace))

        assert len(exprs) == 0

        return processed

    def _do_generate(self, exprs, exclude, cbk_search, cbk_compose=None):
        """
        Carry out the bulk of the work of ``_generate``.
        """
        counter = generator()
        make = lambda: Symbol(name='dummy%d' % counter())

        if cbk_compose is None:
            cbk_compose = lambda *args: None

        mapper = Uxmapper()
        for e in exprs:
            for i in cbk_search(e):
                if not i.is_commutative:
                    continue

                terms = cbk_compose(i)

                # Make sure we won't break any data dependencies
                if terms:
                    free_symbols = set().union(*[i.free_symbols for i in terms])
                else:
                    free_symbols = i.free_symbols
                if {a.function for a in free_symbols} & exclude:
                    continue

                mapper.add(i, make, terms)

        return mapper

    def _generate(self, exprs, exclude):
        """
        Generate one or more extractions from ``exprs``. An extraction is a
        set of CIRE candidates which may be turned into aliases. Two different
        extractions may contain overlapping sub-expressions and, therefore,
        should be processed and evaluated indipendently. An extraction won't
        contain any of the symbols appearing in ``exclude``.
        """
        raise NotImplementedError

    def _lookup_key(self, c):
        """
        Create a key for the given Cluster. Clusters with same key may be
        processed together in the search for CIRE candidates. Clusters should
        have a different key if they must not be processed together, e.g.,
        when this would lead to violation of data dependencies.
        """
        raise NotImplementedError

    def _betterv(self, delta_flops, delta_ws):
        """
        Given the deltas in flop reduction and working set size increase of two
        Variants, return True if the second variant is estimated to deliever
        better performance, False otherwise.
        """
        raise NotImplementedError

    def process(self, clusters):
        raise NotImplementedError


class CireInvariants(CireTransformer, Queue):

    optname = 'invariants'

    def __init__(self, sregistry, options, platform):
        super().__init__(sregistry, options, platform)

        self.opt_maxpar = True
        self.opt_schedule_strategy = None

    def process(self, clusters):
        return self._process_fatd(clusters, 1, xtracted=[])

    def callback(self, clusters, prefix, xtracted=None):
        if not prefix:
            return clusters
        d = prefix[-1].dim

        # Rule out extractions that would break data dependencies
        exclude = set().union(*[c.scope.writes for c in clusters])

        # Rule out extractions that are *not* independent of the Dimension
        # currently investigated
        exclude.add(d)

        key = lambda c: self._lookup_key(c, d)
        processed = list(clusters)
        for ak, group in as_mapper(clusters, key=key).items():
            g = [c for c in group if c.is_dense and c not in xtracted]
            if not g:
                continue

            made = self._aliases_from_clusters(g, exclude, ak)

            if made:
                for n, c in enumerate(g, -len(g)):
                    processed[processed.index(c)] = made.pop(n)
                processed = made + processed

                xtracted.extend(made)

        return processed

    def _generate(self, exprs, exclude):
        # E.g., extract `sin(x)` and `cos(x)` from `a*sin(x)*cos(x)`
        rule = lambda e: e.is_Function or (e.is_Pow and e.exp.is_Number and e.exp < 1)
        cbk_search = lambda e: search(e, rule, 'all', 'bfs_first_hit')
        basextr = self._do_generate(exprs, exclude, cbk_search)
        if not basextr:
            return
        yield basextr

        # E.g., extract `sin(x)*cos(x)` from `a*sin(x)*cos(x)`
        def cbk_search(expr):
            found, others = split(expr.args, lambda a: a in basextr)
            ret = [expr] if found else []
            for a in others:
                ret.extend(cbk_search(a))
            return ret

        cbk_compose = lambda e: split(e.args, lambda a: a in basextr)[0]
        yield self._do_generate(exprs, exclude, cbk_search, cbk_compose)

    def _lookup_key(self, c, d):
        ispace = c.ispace.reset()
        dintervals = c.dspace.intervals.drop(d).reset()
        properties = frozendict({d: relax_properties(v) for d, v in c.properties.items()})

        return AliasKey(ispace, dintervals, c.dtype, None, properties)

    def _betterv(self, delta_flops, delta_ws):
        # Always prefer the Variant with fewer temporaries
        return ((delta_ws == 0 and delta_flops < 0) or
                (delta_ws > 0))


class CireSops(CireTransformer):

    optname = 'sops'

    def __init__(self, sregistry, options, platform):
        super().__init__(sregistry, options, platform)

        self.opt_maxpar = options['cire-maxpar']
        self.opt_schedule_strategy = options['cire-schedule']

    def process(self, clusters):
        processed = []
        for c in clusters:
            if not c.is_dense:
                processed.append(c)
                continue

            # Rule out Dimension-independent dependencies, e.g.:
            # r0 = ...
            # u[x, y] = ... r0*a[x, y] ...
            exclude = {i.source.indexed for i in c.scope.d_flow.independent()}

            # TODO: to process third- and higher-order derivatives, we could
            # extend this by calling `_aliases_from_clusters` repeatedly until
            # `made` is empty. To be investigated
            made = self._aliases_from_clusters([c], exclude, self._lookup_key(c))

            processed.extend(flatten(made) or [c])

        return processed

    def _generate(self, exprs, exclude):
        # E.g., extract `u.dx*a*b` and `u.dx*a*c` from `[(u.dx*a*b).dy`, `(u.dx*a*c).dy]`
        def cbk_search(expr):
            if isinstance(expr, EvalDerivative) and not expr.base.is_Function:
                return expr.args
            else:
                return flatten(e for e in [cbk_search(a) for a in expr.args] if e)

        cbk_compose = lambda e: split_coeff(e)[1]
        basextr = self._do_generate(exprs, exclude, cbk_search, cbk_compose)
        if not basextr:
            return
        yield basextr

        # E.g., extract `u.dx*a` from `[(u.dx*a*b).dy, (u.dx*a*c).dy]`
        # That is, attempt extracting the largest common derivative-induced subexprs
        mappers = [deindexify(e) for e in basextr.extracted]
        counter = Counter(flatten(m.keys() for m in mappers))
        groups = as_mapper(counter, key=counter.get)
        grank = {k: sorted(v, key=lambda e: estimate_cost(e), reverse=True)
                 for k, v in groups.items()}

        def cbk_search2(expr, rank):
            ret = []
            for e in cbk_search(expr):
                mapper = deindexify(e)
                for i in rank:
                    if i in mapper:
                        ret.extend(mapper[i])
                        break
            return ret

        candidates = sorted(grank, reverse=True)[:2]
        for i in candidates:
            lower_pri_elems = flatten([grank[j] for j in candidates if j != i])
            cbk_search_i = lambda e: cbk_search2(e, grank[i] + lower_pri_elems)
            yield self._do_generate(exprs, exclude, cbk_search_i, cbk_compose)

    def _lookup_key(self, c):
        return AliasKey(c.ispace, c.dspace.intervals, c.dtype, c.guards, c.properties)

    def _betterv(self, delta_flops, delta_ws):
        # If there's a greater flop reduction using fewer temporaries, no doubts
        # what's gonna be the best variant. But if the better flop reduction
        # comes at the price of using more temporaries, then we have to apply
        # heuristics, in particular we estimate how many flops would a temporary
        # allow to save
        return ((delta_flops >= 0 and delta_ws > 0 and delta_flops / delta_ws < 15) or
                (delta_flops <= 0 and delta_ws < 0 and delta_flops / delta_ws > 15) or
                (delta_flops <= 0 and delta_ws >= 0))


def collect(extracted, ispace, minstorage):
    """
    Find groups of aliasing expressions.

    We shall introduce the following (loose) terminology:

        * A ``terminal`` is the leaf of a mathematical operation. Terminals
          can be numbers (n), literals (l), or Indexeds (I).
        * ``R`` is the relaxation operator := ``R(n) = n``, ``R(l) = l``,
          ``R(I) = J``, where ``J`` has the same base as ``I`` but with all
          offsets stripped away. For example, ``R(a[i+2,j-1]) = a[i,j]``.
        * A ``relaxed expression`` is an expression in which all of the
          terminals are relaxed.

    Now we define the concept of aliasing. We say that an expression A
    aliases an expression B if:

        * ``R(A) == R(B)``
        * all pairwise Indexeds in A and B access memory locations at a
          fixed constant distance along each Dimension.

    For example, consider the following expressions:

        * a[i+1] + b[i+1]
        * a[i+1] + b[j+1]
        * a[i] + c[i]
        * a[i+2] - b[i+2]
        * a[i+2] + b[i]
        * a[i-1] + b[i-1]

    Out of the expressions above, the following alias to `a[i] + b[i]`:

        * a[i+1] + b[i+1] : same operands and operations, distance along i: 1
        * a[i-1] + b[i-1] : same operands and operations, distance along i: -1

    Whereas the following do not:

        * a[i+1] + b[j+1] : because at least one index differs
        * a[i] + c[i] : because at least one of the operands differs
        * a[i+2] - b[i+2] : because at least one operation differs
        * a[i+2] + b[i] : because the distances along ``i`` differ (+2 and +0)
    """
    # Find the potential aliases
    found = []
    for expr in extracted:
        assert not expr.is_Equality

        indexeds = retrieve_indexed(expr)

        bases = []
        offsets = []
        for i in indexeds:
            ii = IterationInstance(i)
            if ii.is_irregular:
                break

            base = []
            offset = []
            for e, ai in zip(ii, ii.aindices):
                if q_constant(e):
                    base.append(e)
                else:
                    base.append(ai)
                    offset.append((ai, e - ai))
            bases.append(tuple(base))
            offsets.append(LabeledVector(offset))

        if not indexeds or len(bases) == len(indexeds):
            found.append(Candidate(expr, ispace, indexeds, bases, offsets))

    # Create groups of aliasing expressions
    mapper = OrderedDict()
    unseen = list(found)
    while unseen:
        c = unseen.pop(0)
        group = [c]
        for u in list(unseen):
            # Is the arithmetic structure of `c` and `u` equivalent ?
            if not compare_ops(c.expr, u.expr):
                continue

            # Is `c` translated w.r.t. `u` ?
            if not c.translated(u):
                continue

            group.append(u)
            unseen.remove(u)
        group = Group(group)

        if minstorage:
            k = group.dimensions_translated
        else:
            k = group.dimensions
        mapper.setdefault(k, []).append(group)

    aliases = AliasList()
    queue = list(mapper.values())
    while queue:
        groups = queue.pop(0)

        while groups:
            # For each Dimension, determine the Minimum Intervals (MI) spanning
            # all of the Groups diameters
            # Example: x's largest_diameter=2  => [x[-2,0], x[-1,1], x[0,2]]
            # Note: Groups that cannot evaluate their diameter are dropped
            mapper = defaultdict(int)
            for g in list(groups):
                try:
                    mapper.update({d: max(mapper[d], v) for d, v in g.diameter.items()})
                except ValueError:
                    groups.remove(g)
            intervalss = {d: make_rotations_table(d, v) for d, v in mapper.items()}

            # For each Group, find a rotation that is compatible with a given MI
            mapper = {}
            for d, intervals in intervalss.items():
                # Not all groups may access all dimensions
                # Example: `d=t` and groups=[Group(...[t, x]...), Group(...[time, x]...)]
                impacted = [g for g in groups if d in g.dimensions]

                for interval in list(intervals):
                    found = {g: g.find_rotation_distance(d, interval) for g in impacted}
                    if all(distance is not None for distance in found.values()):
                        # `interval` is OK !
                        mapper[interval] = found
                        break

            if len(mapper) == len(intervalss):
                break

            # Try again with fewer groups
            # Heuristic: first try retaining the larger ones
            smallest = len(min(groups, key=len))
            fallback = groups
            groups, remainder = split(groups, lambda g: len(g) > smallest)
            if groups:
                queue.append(remainder)
            elif len(remainder) > 1:
                # No luck with the heuristic, e.g. there are two groups
                # and both have same `len`
                queue.append(fallback[1:])
                groups = [fallback.pop(0)]
            else:
                break

        for g in groups:
            c = g.pivot
            distances = defaultdict(int, [(i.dim, v.get(g)) for i, v in mapper.items()])

            # Create the basis alias
            offsets = [LabeledVector([(l, v[l] + distances[l]) for l in v.labels])
                       for v in c.offsets]
            subs = {i: i.function[[l + v.fromlabel(l, 0) for l in b]]
                    for i, b, v in zip(c.indexeds, c.bases, offsets)}
            pivot = uxreplace(c.expr, subs)

            # All aliased expressions
            aliaseds = [extracted[i.expr] for i in g]

            # Distance of each aliased expression from the basis alias
            distances = []
            for i in g:
                distance = [o.distance(v) for o, v in zip(i.offsets, offsets)]
                distance = [(d, set(v)) for d, v in LabeledVector.transpose(*distance)]
                distances.append(LabeledVector([(d, v.pop()) for d, v in distance]))

            # Compute the alias score
            na = len(aliaseds)
            nr = nredundants(ispace, pivot)
            score = estimate_cost(pivot, True)*((na - 1) + nr)
            if score > 0:
                aliases.add(pivot, aliaseds, list(mapper), distances, score)

    return aliases


def choose(aliases, exprs, mapper, mingain):
    """
    Analyze the detected aliases and, after applying a cost model to rule out
    the aliases with a bad memory/flops trade-off, inject them into the original
    expressions.
    """
    aliases = AliasList(aliases)

    # `score < m` => discarded
    # `score > M` => optimized
    # `m <= score <= M` => maybe discarded, maybe optimized -- depends on heuristics
    m = mingain
    M = mingain*3

    if not aliases:
        return exprs, aliases

    # Filter off the aliases with low score
    key = lambda a: a.score >= m
    aliases.filter(key)

    # Project the candidate aliases into `exprs` to derive the final working set
    mapper = {k: v for k, v in mapper.items() if v.free_symbols & set(aliases.aliaseds)}
    templated = [uxreplace(e, mapper) for e in exprs]
    owset = wset(templated)

    # Filter off the aliases with a weak flop-reduction / working-set tradeoff
    key = lambda a: \
        a.score > M or \
        m <= a.score <= M and max(len(wset(a.pivot)), 1) > len(wset(a.pivot) & owset)
    aliases.filter(key)

    if not aliases:
        return exprs, aliases

    # Substitute the chosen aliasing sub-expressions
    mapper = {k: v for k, v in mapper.items() if v.free_symbols & set(aliases.aliaseds)}
    exprs = [uxreplace(e, mapper) for e in exprs]

    return exprs, aliases


def lower_aliases(aliases, meta, maxpar):
    """
    Create a Schedule from an AliasList.
    """
    dmapper = {}
    processed = []
    for a in aliases:
        imapper = {**{i.dim: i for i in a.intervals},
                   **{i.dim.parent: i for i in a.intervals if i.dim.is_NonlinearDerived}}

        intervals = []
        writeto = []
        sub_iterators = {}
        indicess = [[] for _ in a.distances]
        for i in meta.ispace:
            try:
                interval = imapper[i.dim]
            except KeyError:
                if i.dim in a.free_symbols:
                    # Special case: the Dimension appears within the alias but
                    # not as an Indexed index. Then, it needs to be addeed to
                    # the `writeto` region too
                    interval = i
                else:
                    # E.g., `x0_blk0` or (`a[y_m+1]` => `y not in imapper`)
                    intervals.append(i)
                    continue

            assert i.stamp >= interval.stamp

            if not (writeto or
                    interval != interval.zero() or
                    (maxpar and SEQUENTIAL not in meta.properties.get(i.dim))):
                # The alias doesn't require a temporary Dimension along i.dim
                intervals.append(i)
                continue

            assert not i.dim.is_NonlinearDerived

            # `i.dim` is necessarily part of the write-to region, so
            # we have to adjust the Interval's stamp. For example, consider
            # `i=x[0,0]<1>` and `interval=x[-4,4]<0>`; here we need to
            # use `<1>` as stamp, which is what appears in `ispace`
            interval = interval.lift(i.stamp)

            # We further bump the interval stamp if we were requested to trade
            # fusion for more collapse-parallelism
            interval = interval.lift(interval.stamp + int(maxpar))

            writeto.append(interval)
            intervals.append(interval)

            if i.dim.is_Incr:
                # Suitable IncrDimensions must be used to avoid OOB accesses.
                # E.g., r[xs][ys][z] => both `xs` and `ys` must be initialized such
                # that all accesses are within bounds. This requires traversing the
                # hierarchy of IncrDimensions to set `xs` (`ys`) in a way that
                # consecutive blocks access consecutive regions in `r` (e.g.,
                # `xs=x0_blk1-x0_blk0` with `blocklevels=2`; `xs=0` with
                # `blocklevels=1`, that is it degenerates in this case)
                try:
                    d = dmapper[i.dim]
                except KeyError:
                    dd = i.dim.parent
                    assert dd.is_Incr
                    if dd.parent.is_Incr:
                        # An IncrDimension in between IncrDimensions
                        m = i.dim.symbolic_min - i.dim.parent.symbolic_min
                    else:
                        m = 0
                    d = dmapper[i.dim] = IncrDimension("%ss" % i.dim.name, i.dim, m,
                                                       dd.symbolic_size, 1, dd.step)
                sub_iterators[i.dim] = d
            else:
                d = i.dim

            # Given the iteration `interval`, lower distances to indices
            for distance, indices in zip(a.distances, indicess):
                try:
                    indices.append(d - interval.lower + distance[interval.dim])
                except TypeError:
                    indices.append(d)

        # The alias write-to space
        writeto = IterationSpace(IntervalGroup(writeto), sub_iterators)

        # The alias iteration space
        ispace = IterationSpace(IntervalGroup(intervals, meta.ispace.relations),
                                meta.ispace.sub_iterators,
                                meta.ispace.directions)
        ispace = ispace.augment(sub_iterators)

        processed.append(ScheduledAlias(a.pivot, writeto, ispace, a.aliaseds,
                                        indicess, a.score))

    # The [ScheduledAliases] must be ordered so as to reuse as many of the
    # `ispace`'s IterationIntervals as possible in order to honor the
    # write-to region. Another fundamental reason for ordering is to ensure
    # deterministic code generation
    processed = sorted(processed, key=lambda i: cit(meta.ispace, i.ispace))

    return Schedule(*processed, dmapper=dmapper)


def optimize_schedule_rotations(schedule, sregistry):
    """
    Transform the schedule such that the tensor temporaries "rotate" along
    the outermost Dimension. This trades a parallel Dimension for a smaller
    working set size.
    """
    # The rotations Dimension is the outermost
    ridx = 0

    rmapper = defaultdict(list)
    processed = []
    for k, group in groupby(schedule, key=lambda i: i.writeto):
        g = list(group)

        try:
            candidate = k[ridx]
        except IndexError:
            # Degenerate alias (a scalar)
            processed.extend(g)
            continue
        d = candidate.dim

        try:
            ds = schedule.dmapper[d]
        except KeyError:
            # Can't do anything if `d` isn't an IncrDimension over a block
            processed.extend(g)
            continue

        n = candidate.min_size
        assert n > 0

        iis = candidate.lower
        iib = candidate.upper

        ii = ModuloDimension('%sii' % d, ds, iis, incr=iib)
        cd = CustomDimension(name='%s%s' % (d, d), symbolic_min=ii, symbolic_max=iib,
                             symbolic_size=n)
        dsi = ModuloDimension('%si' % ds, cd, cd + ds - iis, n)

        mapper = OrderedDict()
        for i in g:
            # Update `indicess` to use `xs0`, `xs1`, ...
            mds = []
            for indices in i.indicess:
                v = indices[ridx]
                try:
                    md = mapper[v]
                except KeyError:
                    name = sregistry.make_name(prefix='%sr' % d.name)
                    md = mapper.setdefault(v, ModuloDimension(name, ds, v, n))
                mds.append(md)
            indicess = [indices[:ridx] + [md] + indices[ridx + 1:]
                        for md, indices in zip(mds, i.indicess)]

            # Update `writeto` by switching `d` to `dsi`
            intervals = k.intervals.switch(d, dsi).zero(dsi)
            sub_iterators = dict(k.sub_iterators)
            sub_iterators[d] = dsi
            writeto = IterationSpace(intervals, sub_iterators)

            # Transform `alias` by adding `i`
            pivot = i.pivot.xreplace({d: d + cd})

            # Extend `ispace` to iterate over rotations
            d1 = writeto[ridx+1].dim  # Note: we're by construction in-bounds here
            intervals = IntervalGroup(Interval(cd, 0, 0), relations={(d, cd, d1)})
            rispace = IterationSpace(intervals, {cd: dsi}, {cd: Forward})
            aispace = i.ispace.zero(d)
            aispace = aispace.augment({d: mds + [ii]})
            ispace = IterationSpace.union(rispace, aispace)

            processed.append(ScheduledAlias(pivot, writeto, ispace, i.aliaseds,
                                            indicess, i.score))

        # Update the rotations mapper
        rmapper[d].extend(list(mapper.values()))

    return Schedule(*processed, dmapper=schedule.dmapper, rmapper=rmapper)


def optimize_schedule_padding(schedule, meta, platform):
    """
    Round up the innermost IterationInterval of the tensor temporaries IterationSpace
    to a multiple of the SIMD vector length. This is not always possible though (it
    depends on how much halo is safely accessible in all read Functions).
    """
    processed = []
    for i in schedule:
        try:
            it = i.ispace.itintervals[-1]
            if it.dim is i.writeto[-1].dim and ROUNDABLE in meta.properties[it.dim]:
                vl = platform.simd_items_per_reg(meta.dtype)
                ispace = i.ispace.add(Interval(it.dim, 0, it.interval.size % vl))
            else:
                ispace = i.ispace
            processed.append(ScheduledAlias(i.pivot, i.writeto, ispace, i.aliaseds,
                                            i.indicess, i.score))
        except (TypeError, KeyError, IndexError):
            processed.append(i)

    return Schedule(*processed, dmapper=schedule.dmapper, rmapper=schedule.rmapper)


def lower_schedule(schedule, meta, sregistry, ftemps):
    """
    Turn a Schedule into a sequence of Clusters.
    """
    if ftemps:
        make = TempFunction
    else:
        # Typical case -- the user does *not* "see" the CIRE-created temporaries
        make = Array

    clusters = []
    subs = {}
    for pivot, writeto, ispace, aliaseds, indicess, _ in schedule:
        name = sregistry.make_name()
        dtype = meta.dtype

        if writeto:
            # The Dimensions defining the shape of Array
            # Note: with SubDimensions, we may have the following situation:
            #
            # for zi = z_m + zi_ltkn; zi <= z_M - zi_rtkn; ...
            #   r[zi] = ...
            #
            # Instead of `r[zi - z_m - zi_ltkn]` we have just `r[zi]`, so we'll need
            # as much room as in `zi`'s parent to avoid going OOB
            # Aside from ugly generated code, the reason we do not rather shift the
            # indices is that it prevents future passes to transform the loop bounds
            # (e.g., MPI's comp/comm overlap does that)
            dimensions = [d.parent if d.is_Sub else d for d in writeto.itdimensions]

            # The halo must be set according to the size of writeto space
            halo = [(abs(i.lower), abs(i.upper)) for i in writeto]

            # The indices used to write into the Array
            indices = []
            for i in writeto:
                try:
                    # E.g., `xs`
                    sub_iterators = writeto.sub_iterators[i.dim]
                    assert len(sub_iterators) == 1
                    indices.append(sub_iterators[0])
                except KeyError:
                    # E.g., `z` -- a non-shifted Dimension
                    indices.append(i.dim - i.lower)

            obj = make(name=name, dimensions=dimensions, halo=halo, dtype=dtype)
            expression = Eq(obj[indices], uxreplace(pivot, subs))

            callback = lambda idx: obj[idx]
        else:
            # Degenerate case: scalar expression
            assert writeto.size == 0

            obj = Symbol(name=name, dtype=dtype)
            expression = Eq(obj, uxreplace(pivot, subs))

            callback = lambda idx: obj

        # Create the substitution rules for the aliasing expressions
        subs.update({aliased: callback(indices)
                     for aliased, indices in zip(aliaseds, indicess)})

        # Construct the alias DataSpace
        accesses = detect_accesses(expression)
        parts = {k: IntervalGroup(build_intervals(v)).add(ispace.intervals).relaxed
                 for k, v in accesses.items() if k}
        dspace = DataSpace(meta.dintervals, parts)

        # Drop or weaken parallelism if necessary
        properties = dict(meta.properties)
        for d, v in meta.properties.items():
            if any(i.is_Modulo for i in ispace.sub_iterators[d]):
                properties[d] = normalize_properties(v, {SEQUENTIAL})
            elif d not in writeto.dimensions:
                properties[d] = normalize_properties(v, {PARALLEL_IF_PVT}) - {ROUNDABLE}

        # Finally, build the alias Cluster
        clusters.append(Cluster(expression, ispace, dspace, meta.guards, properties))

    return clusters, subs


def pick_best(variants, schedule_strategy, betterv):
    """
    Return the variant with the best trade-off between operation count
    reduction and working set increase. Heuristics may be applied.
    """
    if type(schedule_strategy) == int:
        try:
            return variants[schedule_strategy]
        except IndexError:
            raise ValueError("Illegal schedule strategy %d; accepted `[0, %d]`"
                             % (schedule_strategy, len(variants) - 1))

    best = None
    best_flops_score = None
    best_ws_score = None

    for i in variants:
        i_flops_score = sum(sa.score for sa in i.schedule)

        # The working set score depends on the number and dimensionality of
        # temporaries required by the Schedule
        i_ws_count = Counter([len(sa.writeto) for sa in i.schedule])
        i_ws_score = tuple(i_ws_count[sa + 1] for sa in reversed(range(max(i_ws_count))))
        # TODO: For now, we assume the performance impact of an N-dimensional
        # temporary is always the same regardless of the value N, but this might
        # change in the future
        i_ws_score = sum(i_ws_score)

        if best is None:
            best, best_flops_score, best_ws_score = i, i_flops_score, i_ws_score
            continue

        delta_flops = best_flops_score - i_flops_score
        delta_ws = best_ws_score - i_ws_score
        if betterv(delta_flops, delta_ws):
            best, best_flops_score, best_ws_score = i, i_flops_score, i_ws_score

    return best


# Utilities


class Candidate(object):

    def __init__(self, expr, ispace, indexeds, bases, offsets):
        self.expr = expr
        self.shifts = ispace.intervals
        self.indexeds = indexeds
        self.bases = bases
        self.offsets = offsets

    def __repr__(self):
        return "Candidate(expr=%s)" % self.expr

    def translated(self, other):
        """
        True if ``self`` is translated w.r.t. ``other``, False otherwise.

        Examples
        --------
        Two candidates are translated if their bases are the same and
        their offsets are pairwise translated.

        c := A[i,j] op A[i,j+1]     -> Toffsets = {i: [0,0], j: [0,1]}
        u := A[i+1,j] op A[i+1,j+1] -> Toffsets = {i: [1,1], j: [0,1]}

        Then `c` is translated w.r.t. `u` with distance `{i: 1, j: 0}`
        """
        if len(self.Toffsets) != len(other.Toffsets):
            return False
        if len(self.bases) != len(other.bases):
            return False

        # Check the bases
        if any(b0 != b1 for b0, b1 in zip(self.bases, other.bases)):
            return False

        # Check the offsets
        for (d0, o0), (d1, o1) in zip(self.Toffsets, other.Toffsets):
            if d0 is not d1:
                return False

            distance = set(o0 - o1)
            if len(distance) != 1:
                return False

        return True

    @cached_property
    def Toffsets(self):
        return LabeledVector.transpose(*self.offsets)

    @cached_property
    def dimensions(self):
        return frozenset(i for i, _ in self.Toffsets)


class Group(tuple):

    """
    A collection of aliasing expressions.
    """

    def __repr__(self):
        return "Group(%s)" % ", ".join([str(i) for i in self])

    def find_rotation_distance(self, d, interval):
        """
        The distance from the Group pivot of a rotation along Dimension ``d`` that
        can safely iterate over the ``interval``.
        """
        assert d is interval.dim

        for rotation, distance in self._pivot_legal_rotations[d]:
            # Does `rotation` cover the `interval` ?
            if rotation.union(interval) != rotation:
                continue

            # Infer the `rotation`'s min_intervals from the pivot's
            min_interval = self._pivot_min_intervals[d].translate(-distance)

            # Does the `interval` actually cover the `rotation`'s `min_interval`?
            if interval.union(min_interval) == interval:
                return distance

        return None

    @cached_property
    def Toffsets(self):
        return [LabeledVector.transpose(*i) for i in zip(*[i.offsets for i in self])]

    @cached_property
    def diameter(self):
        """
        The size of the iteration space required to evaluate all aliasing expressions
        in this Group, along each Dimension.
        """
        ret = defaultdict(int)
        for i in self.Toffsets:
            for d, v in i:
                try:
                    distance = int(max(v) - min(v))
                except TypeError:
                    # An entry in `v` has symbolic components, e.g. `x_m + 2`
                    if len(set(v)) == 1:
                        continue
                    else:
                        raise ValueError
                ret[d] = max(ret[d], distance)

        return ret

    @property
    def pivot(self):
        """
        A deterministically chosen Candidate for this Group.
        """
        return self[0]

    @property
    def dimensions(self):
        return self.pivot.dimensions

    @property
    def dimensions_translated(self):
        return frozenset(d for d, v in self.diameter.items() if v > 0)

    @cached_property
    def _pivot_legal_rotations(self):
        """
        All legal rotations along each Dimension for the Group pivot.
        """
        ret = {}
        for d, (maxd, mini) in self._pivot_legal_shifts.items():
            # Rotation size = mini (min-increment) - maxd (max-decrement)
            v = mini - maxd

            # Build the table of all possible rotations
            m = make_rotations_table(d, v)

            distances = []
            for rotation in m:
                # Distance of the rotation `i` from `c`
                distance = maxd - rotation.lower
                assert distance == mini - rotation.upper
                distances.append(distance)

            ret[d] = list(zip(m, distances))

        return ret

    @cached_property
    def _pivot_min_intervals(self):
        """
        The minimum Interval along each Dimension such that by evaluating the
        pivot, all Candidates are evaluated too.
        """
        c = self.pivot

        ret = defaultdict(lambda: [np.inf, -np.inf])
        for i in self:
            distance = [o.distance(v) for o, v in zip(i.offsets, c.offsets)]
            distance = [(d, set(v)) for d, v in LabeledVector.transpose(*distance)]

            for d, v in distance:
                value = v.pop()
                ret[d][0] = min(ret[d][0], value)
                ret[d][1] = max(ret[d][1], value)

        ret = {d: Interval(d, m, M) for d, (m, M) in ret.items()}

        return ret

    @cached_property
    def _pivot_legal_shifts(self):
        """
        The max decrement and min increment along each Dimension such that the
        Group pivot does not go OOB.
        """
        c = self.pivot

        ret = defaultdict(lambda: (-np.inf, np.inf))
        for i, ofs in zip(c.indexeds, c.offsets):
            f = i.function
            for l in ofs.labels:
                # `f`'s cumulative halo size along `l`
                hsize = sum(f._size_halo[l])

                # Any `ofs`'s shift due to non-[0,0] iteration space
                lower, upper = c.shifts[l].offsets

                try:
                    # Assume `ofs[d]` is a number (typical case)
                    maxd = min(0, max(ret[l][0], -ofs[l] - lower))
                    mini = max(0, min(ret[l][1], hsize - ofs[l] - upper))

                    ret[l] = (maxd, mini)
                except TypeError:
                    # E.g., `ofs[d] = x_m - x + 5`
                    ret[l] = (0, 0)

        return ret


AliasKey = namedtuple('AliasKey', 'ispace dintervals dtype guards properties')
Variant = namedtuple('Variant', 'schedule exprs')


class Alias(object):

    def __init__(self, pivot, aliaseds, intervals, distances, score=0):
        self.pivot = pivot
        self.aliaseds = aliaseds
        self.intervals = intervals
        self.distances = distances
        self.score = score

    def __repr__(self):
        return "Alias<<%s>>" % self.pivot

    @property
    def free_symbols(self):
        return self.pivot.free_symbols

    @property
    def naliases(self):
        return len(self.aliaseds)


class AliasList(object):

    def __init__(self, aliases=None):
        if aliases is None:
            self._list = []
        else:
            self._list = list(aliases._list)

    def __repr__(self):
        if self._list:
            return "AliasList<\n  %s\n>" % ",\n  ".join(str(i) for i in self._list)
        else:
            return "<>"

    def __len__(self):
        return self._list.__len__()

    def __iter__(self):
        for i in self._list:
            yield i

    def add(self, pivot, aliaseds, intervals, distances, score=0):
        assert len(aliaseds) == len(distances)
        self._list.append(Alias(pivot, aliaseds, intervals, distances, score))

    def update(self, aliases):
        self._list.extend(aliases)

    def filter(self, key):
        for i in list(self._list):
            if not key(i):
                self._list.remove(i)

    @property
    def aliaseds(self):
        return flatten(i.aliaseds for i in self._list)


ScheduledAlias = namedtuple('SchedAlias', 'pivot writeto ispace aliaseds indicess score')


class Schedule(tuple):

    def __new__(cls, *items, dmapper=None, rmapper=None):
        obj = super(Schedule, cls).__new__(cls, items)
        obj.dmapper = dmapper or {}
        obj.rmapper = rmapper or {}
        return obj


def make_rotations_table(d, v):
    """
    All possible rotations of `range(v+1)`.
    """
    m = np.array([[j-i if j > i else 0 for j in range(v+1)] for i in range(v+1)])
    m = (m - m.T)[::-1, :]

    # Shift the table so that the middle rotation is at the top
    m = np.roll(m, int(-np.floor(v/2)), axis=0)

    # Turn into a more compact representation as a list of Intervals
    m = [Interval(d, min(i), max(i)) for i in m]

    return m


def cit(ispace0, ispace1):
    """
    The Common IterationIntervals of two IterationSpaces.
    """
    found = []
    for it0, it1 in zip(ispace0.itintervals, ispace1.itintervals):
        if it0 == it1:
            found.append(it0)
        else:
            break
    return tuple(found)


def split_coeff(expr):
    """
    Split potential derivative coefficients and arguments into two groups.
    """
    # TODO: Once we'll be able to keep Derivative intact down to this point,
    # we won't probably need this function anymore, because, in essence, what
    # this function is doing is reconstructing prematurely lowered information
    grids = {getattr(i.function, 'grid', None) for i in expr.free_symbols} - {None}
    if len(grids) != 1:
        return [], None
    grid = grids.pop()

    maybe_coeff = []
    others = []
    for a in expr.args:
        indexeds = [i for i in a.free_symbols if i.is_Indexed]
        if all(not set(grid.dimensions) <= set(i.function.dimensions) for i in indexeds):
            maybe_coeff.append(a)
        else:
            others.append(a)

    return maybe_coeff, others


def nredundants(ispace, expr):
    """
    The number of redundant Dimensions in `ispace` for `expr`. A Dimension is
    redundant if it defines an iteration space for `expr` while not appearing
    among its free symbols. Note that the converse isn't generally true: there
    could be a Dimension that does not appear in the free symbols while defining
    a non-redundant iteration space (e.g., an IncrDimension stemming from blocking).
    """
    iterated = {i.dim for i in ispace}
    used = {i for i in expr.free_symbols if i.is_Dimension}

    # "Short" dimensions won't count
    key = lambda d: d.is_Sub and d.local
    iterated = {d for d in iterated if not key(d)}
    used = {d for d in used if not key(d)}

    iterated = {d.root for d in iterated}
    used = {d.root for d in used}

    return len(iterated) - (len(used))


def wset(exprs):
    """
    Extract the working set out of a set of equations.
    """
    return {i.function for i in flatten([e.free_symbols for e in as_tuple(exprs)])
            if i.function.is_AbstractFunction}


def deindexify(expr):
    """
    Strip away Indexed and indices from an expression, turning them into Functions.
    This means that e.g. `deindexify(f[x+2]*3) == deindexify(f[x-4]*3) == f(x)*3`.
    This function returns a mapper that binds the de-indexified sub-expressions to
    the original counterparts.
    """
    return _deindexify(expr)[1]


@singledispatch
def _deindexify(expr):
    args = []
    mapper = defaultdict(list)
    for a in expr.args:
        arg, m = _deindexify(a)
        args.append(arg)
        for k, v in m.items():
            mapper[k].extend(v)

    rexpr = reuse_if_untouched(expr, args)
    if rexpr is not expr:
        mapper[rexpr] = [expr]

    return rexpr, mapper


@_deindexify.register(sympy.Number)
@_deindexify.register(sympy.Symbol)
@_deindexify.register(sympy.Function)
def _(expr):
    return expr, {}


@_deindexify.register(Indexed)
def _(expr):
    return expr.function, {}
