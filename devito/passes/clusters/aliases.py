from collections import Counter, OrderedDict, defaultdict, namedtuple
from functools import partial, singledispatch
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
from devito.symbolics import (Uxmapper, compare_ops, count, estimate_cost, q_constant,
                              q_leaf, rebuild_if_untouched, retrieve_indexed,
                              search, uxreplace)
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
        # [Clusters]_n -> [AliasList]_m
        variants = []
        for g in self._generators:
            exprs = flatten([c.exprs for c in clusters])

            mapper = g.generate(exprs, exclude)

            found = collect(mapper.extracted, meta.ispace,
                            self.opt_minstorage, self.opt_mingain)

            exprs, aliases = choose(found, exprs, mapper)

            if aliases:
                variants.append(SpacePoint(aliases, exprs))

        # [AliasList]_m -> AliasList (s.t. best memory/flops trade-off)
        try:
            aliases, exprs = pick_best(variants)
        except IndexError:
            return []

        # AliasList -> Schedule
        schedule = lower_aliases(aliases, meta, self.opt_maxpar)

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

    @property
    def _generators(self):
        """
        A CireTransformer uses one or more Generators to extract sub-expressions
        that are potential CIRE candidates. Different Generators may capture
        different sets of sub-expressions, and therefore be characterized by
        different memory/flops trade-offs.
        """
        raise NotImplementedError

    def _lookup_key(self, c):
        """
        Create a key for the given Cluster. Clusters with same key may be processed
        together to find redundant aliases. Clusters should have a different key
        if they cannot be processed together, e.g., when this would lead to
        dependencies violation.
        """
        raise NotImplementedError

    def process(self, clusters):
        raise NotImplementedError


class CireInvariants(CireTransformer, Queue):

    optname = 'invariants'

    def __init__(self, sregistry, options, platform):
        super().__init__(sregistry, options, platform)

        self.opt_maxpar = True

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

            made = self._aliases_from_clusters(g, exclude, ak)

            if made:
                for n, c in enumerate(g, -len(g)):
                    processed[processed.index(c)] = made.pop(n)
                processed = made + processed

                xtracted.extend(made)

        return processed

    @property
    def _generators(self):
        return (GeneratorExpensive, GeneratorExpensiveCompounds)

    def _lookup_key(self, c, d):
        ispace = c.ispace.reset()
        dintervals = c.dspace.intervals.drop(d).reset()
        properties = frozendict({d: relax_properties(v) for d, v in c.properties.items()})
        return AliasKey(ispace, dintervals, c.dtype, None, properties)


class CireSops(CireTransformer):

    optname = 'sops'

    def __init__(self, sregistry, options, platform):
        super().__init__(sregistry, options, platform)

        self.opt_maxpar = options['cire-maxpar']

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

            #TODO: TO OPT NESTED DERIVATIVES, JUST KEEP ITERATING UNTIL
            # MADE'S PROCESSED-PART IS EMPTY
            made = self._aliases_from_clusters([c], exclude, self._lookup_key(c))

            processed.extend(flatten(made) or [c])

        return processed

    @property
    def _generators(self):
        return (GeneratorDerivativeCompounds,)
        #return (GeneratorDerivative, GeneratorDerivativeCompounds)

    def _lookup_key(self, c):
        return AliasKey(c.ispace, c.dspace.intervals, c.dtype, c.guards, c.properties)


modes = {
    CireInvariants.optname: CireInvariants,
    CireSops.optname: CireSops,
}


class Generator(object):

    @classmethod
    def _basextr(cls, exprs, exclude, make):
        return Uxmapper()

    @classmethod
    def _search(cls, expr, basextr):
        raise NotImplementedError

    @classmethod
    def _compose(cls, expr, basextr):
        return None

    @classmethod
    def generate(cls, exprs, exclude, make=None):
        if make is None:
            counter = generator()
            make = lambda: Symbol(name='dummy%d' % counter())

        basextr = cls._basextr(exprs, exclude, make)

        mapper = Uxmapper()
        for e in exprs:
            for i in cls._search(e, basextr):
                if not i.is_commutative:
                    continue

                terms = cls._compose(i, basextr)

                # Make sure we won't break any data dependencies
                if terms:
                    free_symbols = set().union(*[i.free_symbols for i in terms])
                else:
                    free_symbols = i.free_symbols
                if {a.function for a in free_symbols} & exclude:
                    continue

                mapper.add(i, make, terms)

        return mapper


class GeneratorExpensive(Generator):

    @classmethod
    def _search(cls, expr, *args):
        rule = lambda e: e.is_Function or (e.is_Pow and e.exp.is_Number and e.exp < 1)
        return search(expr, rule, 'all', 'bfs_first_hit')


class GeneratorExpensiveCompounds(GeneratorExpensive):

    @classmethod
    def _basextr(cls, exprs, exclude, make):
        return cls.__base__.generate(exprs, exclude, make)

    @classmethod
    def _search(cls, expr, basextr):
        rule = lambda e: any(a in basextr for a in e.args)
        return search(expr, rule, 'all', 'dfs')

    @classmethod
    def _compose(cls, expr, basextr):
        return split(expr.args, lambda a: a in basextr)[0]


class GeneratorDerivative(Generator):

    @classmethod
    def _search(cls, expr, *args):
        if isinstance(expr, EvalDerivative) and not expr.base.is_Function:
            return expr.args
        else:
            return flatten(e for e in [cls._search(a) for a in expr.args] if e)

    @classmethod
    def _compose(cls, expr, *args):
        return split_coeff(expr)[1]


class GeneratorDerivativeCompounds(GeneratorDerivative):

    @classmethod
    def _basextr(cls, exprs, exclude, make):
        basextr = cls.__base__.generate(exprs, exclude, make)

        # Find the largest de-indexified sub-expressions redundantly
        # appearing across multiple exprs
        mappers = [deindexify(e) for e in basextr.extracted]

        # Sort by occurrences and costs
        counter = Counter(flatten(m.keys() for m in mappers))
        rank = sorted(counter, key=lambda e: (counter[e], estimate_cost(e)), reverse=True)

        return rank

    @classmethod
    def _search(cls, expr, basextr):
        found = cls.__base__._search(expr)

        ret = []
        for e in found:
            mapper = deindexify(e)
            for i in basextr:
                if i in mapper:
                    ret.extend(mapper[i])
                    break

        return ret


def collect(extracted, ispace, minstorage, mingain):
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

            # Compute the alias score. With a score of 0, the alias is discarded
            na = len(aliaseds)
            nr = nredundants(ispace, pivot)
            try:
                score = estimate_cost(pivot, True)*((na - 1) + nr) // mingain
            except ZeroDivisionError:
                score = np.inf
            if score > 0:
                aliases.add(pivot, aliaseds, list(mapper), distances, score)

    return aliases


def choose(aliases, exprs, mapper):
    """
    Analyze the detected aliases and, after applying a cost model to rule out
    the aliases with a bad memory/flops trade-off, inject them into the original
    expressions.
    """
    aliases = AliasList(aliases)

    if not aliases:
        return exprs, aliases

    # Project the candidate aliases into exprs to determine what the new
    # working set would be
    mapper = {k: v for k, v in mapper.items() if v.free_symbols & set(aliases.aliaseds)}
    templated = [uxreplace(e, mapper) for e in exprs]

    # Filter off the aliases inducing a non favorable tradeoff between operation
    # count reduction and working set size increase
    owset = wset(templated)
    key = lambda a: \
        a.score > 2 or \
        1 <= a.score <= 2 and max(len(wset(a.pivot)), 1) > len(wset(a.pivot) & owset)
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

            if not (writeto or interval != interval.zero() or maxpar):
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

        processed.append(ScheduledAlias(a.pivot, writeto, ispace, a.aliaseds, indicess))

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

            processed.append(ScheduledAlias(pivot, writeto, ispace, i.aliaseds, indicess))

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
            if it.dim is i.writeto[-1].dim and  ROUNDABLE in meta.properties[it.dim]:
                vl = platform.simd_items_per_reg(meta.dtype)
                ispace = i.ispace.add(Interval(it.dim, 0, it.interval.size % vl))
            else:
                ispace = i.ispace
            processed.append(ScheduledAlias(i.pivot, i.writeto, ispace, i.aliaseds,
                                            i.indicess))
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
    for pivot, writeto, ispace, aliaseds, indicess in schedule:
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


def pick_best(variants):
    """
    Use the variant score and heuristics to return the variant with the best
    trade-off between operation count reduction and working set increase.
    """
    best = variants.pop(0)
    for i in variants:
        best_flop_score, best_ws_score = best.aliases.score
        if best_flop_score == 0:
            best = i
            continue

        i_flop_score, i_ws_score = i.aliases.score

        # The current heustic is fairly basic: the one with smaller working
        # set size increase wins, unless there's a massive reduction in operation
        # count in the other one
        delta = i_ws_score - best_ws_score
        if (delta > 0 and i_flop_score / best_flop_score > 100) or \
           (delta == 0 and i_flop_score > best_flop_score) or \
           (delta < 0 and best_flop_score / i_flop_score <= 100):
            best = i

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

ScheduledAlias = namedtuple('ScheduledAlias', 'pivot writeto ispace aliaseds indicess')
ScheduledAlias.__new__.__defaults__ = (None,) * len(ScheduledAlias._fields)

SpacePoint = namedtuple('SpacePoint', 'aliases exprs')


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
    def is_scalar(self):
        return not any(i.is_Dimension for i in self.free_symbols)

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

    @property
    def score(self):
        # The score is a 2-tuple <flop-reduction-score, workin-set-score>
        flop_score = sum(i.score for i in self._list)
        ws_score = len([i for i in self if not i.is_scalar])
        return flop_score, ws_score


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
    # we won't need neither `split_coeff` nor `maybe_coeff` anymore
    grids = {getattr(i.function, 'grid', None) for i in expr.free_symbols} - {None}
    if len(grids) != 1:
        return [], None
    grid = grids.pop()
    return split(expr.args, partial(maybe_coeff, grid))


def maybe_coeff(grid, expr):
    """
    True if `expr` could be the coefficient of an FD derivative, False otherwise.
    """
    if expr.is_Number:
        return True
    indexeds = [i for i in expr.free_symbols if i.is_Indexed]
    if not indexeds:
        return True
    return any(not set(grid.dimensions) <= set(i.function.dimensions) for i in indexeds)


def nredundants(ispace, expr):
    """
    The number of redundant Dimensions in `ispace` for `expr`. A Dimension is
    redundant if it defines an iteration space for `expr` while not appearing
    among its free symbols. Note that the converse isn't generally true: there
    could be a Dimension that does not appear in the free symbols while defining
    a non-redundant iteration space (e.g., an IncrDimension stemming from blocking).
    """
    return (len({i.dim.root for i in ispace}) -
            len({i.root for i in expr.free_symbols if i.is_Dimension}))


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

    rexpr = rebuild_if_untouched(expr, args)
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
