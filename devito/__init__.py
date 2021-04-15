from itertools import product

# Import the global `configuration` dict
from devito.parameters import *  # noqa

# DSL imports
from devito.types import NODE, CELL, Buffer  # noqa
from devito.types.caching import _SymbolCache, CacheManager  # noqa
from devito.types.constant import *  # noqa
from devito.types.dimension import *  # noqa
from devito.types.dense import *  # noqa
from devito.types.equation import *  # noqa
from devito.types.grid import *  # noqa
from devito.types.relational import *  # noqa
from devito.types.sparse import *  # noqa
from devito.types.tensor import *  # noqa
from devito.finite_differences import *  # noqa
from devito.operations.solve import *
from devito.operator import Operator  # noqa

# Other stuff exposed to the user
from devito.builtins import *  # noqa
from devito.data.allocators import *  # noqa
from devito.mpi import MPI, mpi_registry  # noqa

# Imports required to initialize Devito
from devito.arch import compiler_registry, platform_registry
from devito.core import *   # noqa
from devito.logger import error, warning, info, logger_registry, set_log_level  # noqa
from devito.operator import profiler_registry, operator_registry


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions


def reinit_compiler(val):
    """
    Re-initialize the Compiler.
    """
    configuration['compiler'].__init__(suffix=configuration['compiler'].suffix,
                                       mpi=configuration['mpi'])


# Setup target platform and compiler
configuration.add('platform', 'cpu64', list(platform_registry),
                  callback=lambda i: platform_registry[i]())
configuration.add('compiler', 'custom', list(compiler_registry),
                  callback=lambda i: compiler_registry[i]())

# Setup language for shared-memory parallelism
preprocessor = lambda i: {0: 'C', 1: 'openmp'}.get(i, i)  # Handles DEVITO_OPENMP deprec
configuration.add('language', 'C', [0, 1, 'C', 'openmp', 'openacc'],
                  preprocessor=preprocessor, callback=reinit_compiler, deprecate='openmp')

# MPI mode (0 => disabled, 1 == basic)
preprocessor = lambda i: bool(i) if isinstance(i, int) else i
configuration.add('mpi', 0, [0, 1] + list(mpi_registry),
                  preprocessor=preprocessor, callback=reinit_compiler)

# Should Devito run a first-touch Operator upon data allocation?
configuration.add('first-touch', 0, [0, 1], preprocessor=bool, impacts_jit=False)

# Should Devito ignore any unknown runtime arguments supplied to Operator.apply(),
# or rather raise an exception (the default behaviour)?
configuration.add('ignore-unknowns', 0, [0, 1], preprocessor=bool, impacts_jit=False)

# Setup log level
configuration.add('log-level', 'INFO', list(logger_registry),
                  callback=set_log_level, impacts_jit=False)

# Escape hatch for custom kernels. The typical use case is as follows: one lets
# Devito generate code for an Operator; then, once the session is over, the
# generated file is manually modified (e.g., for debugging or for performance
# experimentation); finally, when re-running the same program, Devito won't
# overwrite the user-modified files (thus entirely bypassing code generation),
# and will instead use the custom kernel
configuration.add('jit-backdoor', 0, [0, 1], preprocessor=bool, impacts_jit=False)

# By default unsafe math is allowed as most applications are insensitive to
# floating-point roundoff errors. Enabling this disables unsafe math
# optimisations.
configuration.add('safe-math', 0, [0, 1], preprocessor=bool, callback=reinit_compiler)

# Enable/disable automatic padding for allocated data
configuration.add('autopadding', False, [False, True])


def autotune_callback(val):  # noqa
    if isinstance(val, str):
        level, mode = val, 'preemptive'  # default mode
    else:
        level, mode = val
    if level == 'off':
        level = False
    return (level, mode)


# Setup autotuning
levels = ['off', 'basic', 'aggressive', 'max']
modes = ['preemptive', 'destructive', 'runtime']
accepted = levels + [list(i) for i in product(levels, modes)]
configuration.add('autotuning', 'off', accepted, callback=autotune_callback,
                  impacts_jit=False)

# In develop-mode:
# - Some optimizations may not be applied to the generated code.
# - The compiler performs more type and value checking
configuration.add('develop-mode', True, [False, True])

# Setup optimization level
configuration.add('opt', 'advanced', list(operator_registry._accepted), deprecate='dle')
configuration.add('opt-options', {}, deprecate='dle-options')

# Setup Operator profiling
configuration.add('profiling', 'basic', list(profiler_registry), impacts_jit=False)

# Initialize `configuration`
init_configuration()

# Expose a mechanism to clean up the symbol caches (SymPy's, Devito's)
clear_cache = CacheManager().clear  # noqa


# Helper functions to switch on/off optimisation levels
def mode_develop():
    """Run all future Operators in develop mode. This is the default mode."""
    configuration['develop-mode'] = True


def mode_performance():
    """
    Run all future Operators in performance mode. The performance mode
    also employs suitable NUMA strategies for memory allocation.
    """
    configuration['develop-mode'] = False
    configuration['autotuning'] = 'aggressive'
    # With the autotuner in `aggressive` mode, a more aggressive blocking strategy
    # which also tiles the innermost loop) is beneficial
    configuration['opt-options']['blockinner'] = True
