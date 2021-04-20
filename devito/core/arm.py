from devito.core.cpu import Cpu64AdvOperator
from devito.passes.iet import (CTarget, OmpTarget, mpiize, optimize_halospots,
                               hoist_prodders, relax_incr_dimensions)
from devito.tools import timed_pass

__all__ = ['ArmAdvCOperator', 'ArmAdvOmpOperator']


class ArmAdvOperator(Cpu64AdvOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # Lower IncrDimensions so that blocks of arbitrary shape may be used
        relax_incr_dimensions(graph)

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


class ArmAdvCOperator(ArmAdvOperator):
    _Target = CTarget


class ArmAdvOmpOperator(ArmAdvOperator):
    _Target = OmpTarget

    # Avoid nested parallelism on ThunderX2
    PAR_NESTED = 4
