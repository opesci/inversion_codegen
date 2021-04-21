from functools import partial

import numpy as np

from devito.core.operator import CoreOperator, CustomOperator
from devito.exceptions import InvalidOperator
from devito.passes.equations import buffering, collect_derivatives
from devito.passes.clusters import (Lift, blocking, cire, cse, extract_increments,
                                    factorize, fuse, optimize_pows)
from devito.passes.iet import (CTarget, OmpTarget, avoid_denormals, mpiize,
                               optimize_halospots, hoist_prodders, relax_incr_dimensions)
from devito.tools import timed_pass

__all__ = ['Cpu64NoopCOperator', 'Cpu64NoopOmpOperator', 'Cpu64AdvCOperator',
           'Cpu64AdvOmpOperator', 'Cpu64FsgCOperator', 'Cpu64FsgOmpOperator',
           'Cpu64CustomOperator']


class Cpu64OperatorMixin(object):

    BLOCK_LEVELS = 1
    """
    Loop blocking depth. So, 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    CIRE_MINGAIN = 10
    """
    Minimum operation count reduction for a redundant expression to be optimized
    away. Higher (lower) values make a redundant expression less (more) likely to
    be optimized away.
    """

    PAR_COLLAPSE_NCORES = 4
    """
    Use a collapse clause if the number of available physical cores is greater
    than this threshold.
    """

    PAR_COLLAPSE_WORK = 100
    """
    Use a collapse clause if the trip count of the collapsable loops is statically
    known to exceed this threshold.
    """

    PAR_CHUNK_NONAFFINE = 3
    """
    Coefficient to adjust the chunk size in non-affine parallel loops.
    """

    PAR_DYNAMIC_WORK = 10
    """
    Use dynamic scheduling if the operation count per iteration exceeds this
    threshold. Otherwise, use static scheduling.
    """

    PAR_NESTED = 2
    """
    Use nested parallelism if the number of hyperthreads per core is greater
    than this threshold.
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        o = {}
        oo = kwargs['options']

        # Execution modes
        o['openmp'] = oo.pop('openmp')
        o['mpi'] = oo.pop('mpi')
        o['parallel'] = o['openmp']  # Backwards compatibility

        # Buffering
        o['buf-async-degree'] = oo.pop('buf-async-degree', None)

        # Blocking
        o['blockinner'] = oo.pop('blockinner', False)
        o['blocklevels'] = oo.pop('blocklevels', cls.BLOCK_LEVELS)
        o['skewing'] = oo.pop('skewing', False)

        # CIRE
        o['min-storage'] = oo.pop('min-storage', False)
        o['cire-rotate'] = oo.pop('cire-rotate', False)
        o['cire-maxpar'] = oo.pop('cire-maxpar', False)
        o['cire-ftemps'] = oo.pop('cire-ftemps', False)
        o['cire-mingain'] = oo.pop('cire-mingain', cls.CIRE_MINGAIN)

        # Shared-memory parallelism
        o['par-collapse-ncores'] = oo.pop('par-collapse-ncores', cls.PAR_COLLAPSE_NCORES)
        o['par-collapse-work'] = oo.pop('par-collapse-work', cls.PAR_COLLAPSE_WORK)
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = oo.pop('par-dynamic-work', cls.PAR_DYNAMIC_WORK)
        o['par-nested'] = oo.pop('par-nested', cls.PAR_NESTED)

        # Recognised but unused by the CPU backend
        oo.pop('par-disabled', None)
        oo.pop('gpu-direct', None)
        oo.pop('gpu-fit', None)

        if oo:
            raise InvalidOperator("Unrecognized optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs


# Mode level


class Cpu64NoopOperator(Cpu64OperatorMixin, CoreOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Shared-memory parallelism
        if options['openmp']:
            parizer = cls._Target.Parizer(sregistry, options, platform)
            parizer.make_parallel(graph)
            parizer.initialize(graph)

        # Symbol definitions
        cls._Target.DataManager(sregistry).process(graph)

        return graph


class Cpu64AdvOperator(Cpu64OperatorMixin, CoreOperator):

    @classmethod
    @timed_pass(name='specializing.DSL')
    def _specialize_dsl(cls, expressions, **kwargs):
        expressions = collect_derivatives(expressions)

        return expressions

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = fuse(clusters, toposort=True)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, 'invariants', sregistry, options, platform)
        clusters = Lift().process(clusters)

        # Blocking to improve data locality
        clusters = blocking(clusters, options)

        # Reduce flops
        clusters = extract_increments(clusters, sregistry)
        clusters = cire(clusters, 'sops', sregistry, options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)

        # The previous passes may have created fusion opportunities
        clusters = fuse(clusters)

        # Reduce flops
        clusters = cse(clusters, sregistry)

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Flush denormal numbers
        avoid_denormals(graph)

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph, sregistry=sregistry)

        # Parallelism
        parizer = cls._Target.Parizer(sregistry, options, platform)
        parizer.make_simd(graph)
        parizer.make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        cls._Target.DataManager(sregistry).process(graph)

        # Initialize the target-language runtime
        parizer.initialize(graph)

        return graph


class Cpu64FsgOperator(Cpu64AdvOperator):

    """
    Operator with performance optimizations tailored "For small grids" ("Fsg").
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        kwargs = super()._normalize_kwargs(**kwargs)

        if kwargs['options']['min-storage']:
            raise InvalidOperator('You should not use `min-storage` with `advanced-fsg '
                                  ' as they work in opposite directions')

        return kwargs

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = fuse(clusters, toposort=True)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, 'invariants', sregistry, options, platform)
        clusters = Lift().process(clusters)

        # Reduce flops (potential arithmetic alterations)
        clusters = extract_increments(clusters, sregistry)
        clusters = cire(clusters, 'sops', sregistry, options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)

        # The previous passes may have created fusion opportunities
        clusters = fuse(clusters)

        # Reduce flops (no arithmetic alterations)
        clusters = cse(clusters, sregistry)

        # Blocking to improve data locality
        clusters = blocking(clusters, options)

        return clusters


class Cpu64CustomOperator(Cpu64OperatorMixin, CustomOperator):

    _Target = OmpTarget

    @classmethod
    def _make_dsl_passes_mapper(cls, **kwargs):
        return {
            'collect-derivs': collect_derivatives,
        }

    @classmethod
    def _make_exprs_passes_mapper(cls, **kwargs):
        options = kwargs['options']

        # This callback simply mimics `is_on_device`, used in the device backends.
        # It's used by `buffering` to replace `save!=None` TimeFunctions with buffers
        def callback(f):
            if f.is_TimeFunction and f.save is not None:
                return [f.time_dim]
            else:
                return None

        return {
            'buffering': lambda i: buffering(i, callback, options)
        }

    @classmethod
    def _make_clusters_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        return {
            'blocking': lambda i: blocking(i, options),
            'factorize': factorize,
            'fuse': fuse,
            'lift': lambda i: Lift().process(cire(i, 'invariants', sregistry,
                                                  options, platform)),
            'cire-sops': lambda i: cire(i, 'sops', sregistry, options, platform),
            'cse': lambda i: cse(i, sregistry),
            'opt-pows': optimize_pows,
            'topofuse': lambda i: fuse(i, toposort=True)
        }

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        parizer = cls._Target.Parizer(sregistry, options, platform)

        return {
            'denormals': avoid_denormals,
            'optcomms': optimize_halospots,
            'blocking': partial(relax_incr_dimensions, sregistry=sregistry),
            'parallel': parizer.make_parallel,
            'openmp': parizer.make_parallel,
            'mpi': partial(mpiize, mode=options['mpi']),
            'simd': partial(parizer.make_simd),
            'prodders': hoist_prodders,
            'init': parizer.initialize
        }

    _known_passes = (
        # DSL
        'collect-derivs',
        # Expressions
        'buffering',
        # Clusters
        'blocking', 'topofuse', 'fuse', 'factorize', 'cire-sops', 'cse', 'lift',
        'opt-pows',
        # IET
        'denormals', 'optcomms', 'openmp', 'mpi', 'simd', 'prodders',
    )
    _known_passes_disabled = ('tasking', 'streaming', 'gpu-direct', 'openacc')
    assert not (set(_known_passes) & set(_known_passes_disabled))


# Language level


class Cpu64NoopCOperator(Cpu64NoopOperator):
    _Target = CTarget


class Cpu64NoopOmpOperator(Cpu64NoopOperator):
    _Target = OmpTarget


class Cpu64AdvCOperator(Cpu64AdvOperator):
    _Target = CTarget


class Cpu64AdvOmpOperator(Cpu64AdvOperator):
    _Target = OmpTarget


class Cpu64FsgCOperator(Cpu64FsgOperator):
    _Target = CTarget


class Cpu64FsgOmpOperator(Cpu64FsgOperator):
    _Target = OmpTarget
