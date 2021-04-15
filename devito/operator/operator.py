from collections import OrderedDict
from functools import reduce
from operator import attrgetter, mul
from math import ceil

from cached_property import cached_property
import ctypes

from devito.arch import compiler_registry, platform_registry
from devito.exceptions import InvalidOperator
from devito.logger import info, perf, warning, is_log_enabled_for
from devito.ir.equations import LoweredEq, lower_exprs, generate_implicit_exprs
from devito.ir.clusters import ClusterGroup, clusterize
from devito.ir.iet import Callable, EntryFunction, MetaCall, derive_parameters, iet_build
from devito.ir.stree import stree_build
from devito.operator.profiling import create_profile
from devito.operator.registry import operator_selector
from devito.operator.symbols import SymbolRegistry
from devito.mpi import MPI
from devito.parameters import configuration
from devito.passes import Graph, instrument
from devito.symbolics import estimate_cost
from devito.tools import (DAG, Signer, ReducerMap, as_tuple, flatten, filter_ordered,
                          filter_sorted, split, timed_pass, timed_region)
from devito.types import Evaluable

__all__ = ['Operator']


class Operator(Callable):

    """
    Generate, JIT-compile and run C code starting from an ordered sequence
    of symbolic expressions.

    Parameters
    ----------
    expressions : expr-like or list or expr-like
        The (list of) expression(s) defining the Operator computation.
    **kwargs
        * name : str
            Name of the Operator, defaults to "Kernel".
        * subs : dict
            Symbolic substitutions to be applied to ``expressions``.
        * opt : str
            The performance optimization level. Defaults to ``configuration['opt']``.
        * language : str
            The target language for shared-memory parallelism. Defaults to
            ``configuration['language']``.
        * platform : str
            The architecture the code is generated for. Defaults to
            ``configuration['platform']``.
        * compiler : str
            The backend compiler used to jit-compile the generated code.
            Defaults to ``configuration['compiler']``.

    Examples
    --------
    The following Operator implements a trivial time-marching method that
    adds 1 to every grid point in ``u`` at every timestep.

    >>> from devito import Eq, Grid, TimeFunction, Operator
    >>> grid = Grid(shape=(4, 4))
    >>> u = TimeFunction(name='u', grid=grid)
    >>> op = Operator(Eq(u.forward, u + 1))

    Multiple expressions can be supplied, and there is no limit to the number of
    expressions in an Operator.

    >>> v = TimeFunction(name='v', grid=grid)
    >>> op = Operator([Eq(u.forward, u + 1),
    ...                Eq(v.forward, v + 1)])

    Simple boundary conditions can be imposed easily exploiting the "indexed
    notation" for Functions/TimeFunctions.

    >>> t = grid.stepping_dim
    >>> x, y = grid.dimensions
    >>> op = Operator([Eq(u.forward, u + 1),
    ...                Eq(u[t+1, x, 0], 0),
    ...                Eq(u[t+1, x, 2], 0),
    ...                Eq(u[t+1, 0, y], 0),
    ...                Eq(u[t+1, 2, y], 0)])

    A semantically equivalent computation can be expressed exploiting SubDomains.

    >>> u.data[:] = 0
    >>> op = Operator(Eq(u.forward, u + 1, subdomain=grid.interior))

    By specifying a SubDomain, the Operator constrains the execution of an expression to
    a certain sub-region within the computational domain. Ad-hoc SubDomains can also be
    created in application code -- refer to the SubDomain documentation for more info.

    Advanced boundary conditions can be expressed leveraging `SubDomain` and
    `SubDimension`.

    Tensor contractions are supported, but with one caveat: in case of MPI execution, any
    global reductions along an MPI-distributed Dimension should be handled explicitly in
    user code. The following example shows how to implement the matrix-vector
    multiplication ``Av = b`` (inducing a reduction along ``y``).

    >>> from devito import Inc, Function
    >>> A = Function(name='A', grid=grid)
    >>> v = Function(name='v', shape=(3,), dimensions=(y,))
    >>> b = Function(name='b', shape=(3,), dimensions=(x,))
    >>> op = Operator(Inc(b, A*v))

    Dense and sparse computation may be present within the same Operator. In the
    following example, interpolation is used to approximate the value of four
    sparse points placed at the center of the four quadrants at the grid corners.

    >>> import numpy as np
    >>> from devito import SparseFunction
    >>> grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
    >>> f = Function(name='f', grid=grid)
    >>> coordinates = np.array([(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)])
    >>> sf = SparseFunction(name='sf', grid=grid, npoint=4, coordinates=coordinates)
    >>> op = Operator([Eq(f, f + 1)] + sf.interpolate(f))

    The iteration direction is automatically detected by the Devito compiler. Below,
    the Operator runs from ``time_M`` (maximum point in the time dimension) down to
    ``time_m`` (minimum point in the time dimension), as opposed to all of the examples
    seen so far, in which the execution along time proceeds from ``time_m`` to ``time_M``
    through unit-step increments.

    >>> op = Operator(Eq(u.backward, u + 1))

    Loop-level optimisations, including SIMD vectorisation and OpenMP parallelism, are
    automatically discovered and handled by the Devito compiler. For more information,
    refer to the relevant documentation.
    """

    _default_headers = [('_POSIX_C_SOURCE', '200809L')]
    _default_includes = ['stdlib.h', 'math.h', 'sys/time.h']
    _default_globals = []

    def __new__(cls, expressions, **kwargs):
        if expressions is None:
            # Return a dummy Callable. This is exploited by unpickling. Users
            # can't do anything useful with it
            return super(Operator, cls).__new__(cls, **kwargs)

        # Parse input arguments
        kwargs = parse_kwargs(**kwargs)

        # The Operator type for the given target
        cls = operator_selector(**kwargs)

        # Normalize input arguments for the selected Operator
        kwargs = cls._normalize_kwargs(**kwargs)

        # Create a symbol registry
        kwargs['sregistry'] = SymbolRegistry()

        # Lower to a JIT-compilable object
        with timed_region('op-compile') as r:
            op = cls._build(expressions, **kwargs)
        op._profiler.py_timers.update(r.timings)

        # Emit info about how long it took to perform the lowering
        op._emit_build_profiling()

        return op

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        return kwargs

    @classmethod
    def _build(cls, expressions, **kwargs):
        expressions = as_tuple(expressions)

        # Input check
        if any(not isinstance(i, Evaluable) for i in expressions):
            raise InvalidOperator("Only `devito.Evaluable` are allowed.")

        # Python-level (i.e., compile time) and C-level (i.e., run time) performance
        profiler = create_profile('timers')

        # Lower input expressions
        expressions = cls._lower_exprs(expressions, **kwargs)

        # Group expressions based on iteration spaces and data dependences
        clusters = cls._lower_clusters(expressions, profiler, **kwargs)

        # Lower Clusters to a ScheduleTree
        stree = cls._lower_stree(clusters, **kwargs)

        # Lower ScheduleTree to an Iteration/Expression Tree
        iet, byproduct = cls._lower_iet(stree, profiler, **kwargs)

        # Make it an actual Operator
        op = Callable.__new__(cls, **iet.args)
        Callable.__init__(op, **op.args)

        # Header files, etc.
        op._headers = list(cls._default_headers)
        op._headers.extend(byproduct.headers)
        op._globals = list(cls._default_globals)
        op._includes = list(cls._default_includes)
        op._includes.extend(profiler._default_includes)
        op._includes.extend(byproduct.includes)

        # Required for the jit-compilation
        op._compiler = kwargs['compiler']
        op._lib = None
        op._cfunction = None

        # References to local or external routines
        op._func_table = OrderedDict()
        op._func_table.update(OrderedDict([(i, MetaCall(None, False))
                                           for i in profiler._ext_calls]))
        op._func_table.update(OrderedDict([(i.root.name, i) for i in byproduct.funcs]))

        # Internal state. May be used to store information about previous runs,
        # autotuning reports, etc
        op._state = cls._initialize_state(**kwargs)

        # Produced by the various compilation passes
        op._input = filter_sorted(flatten(e.reads + e.writes for e in expressions))
        op._output = filter_sorted(flatten(e.writes for e in expressions))
        op._dimensions = flatten(c.dimensions for c in clusters) + byproduct.dimensions
        op._dimensions = sorted(set(op._dimensions), key=attrgetter('name'))
        op._dtype, op._dspace = clusters.meta
        op._profiler = profiler

        return op

    def __init__(self, *args, **kwargs):
        # Bypass the silent call to __init__ triggered through the backends engine
        pass

    # Compilation -- Expression level

    @classmethod
    def _initialize_state(cls, **kwargs):
        return {'optimizations': kwargs.get('mode', configuration['opt'])}

    @classmethod
    def _specialize_dsl(cls, expressions, **kwargs):
        """
        Backend hook for specialization at the DSL level. The input is made of
        expressions and other higher order objects such as Injection or
        Interpolation; the expressions are still unevaluated at this stage,
        meaning that they are still in tensorial form and derivatives aren't
        expanded yet.
        """
        return expressions

    @classmethod
    def _specialize_exprs(cls, expressions, **kwargs):
        """
        Backend hook for specialization at the expression level.
        """
        return expressions

    @classmethod
    @timed_pass(name='lowering.Expressions')
    def _lower_exprs(cls, expressions, **kwargs):
        """
        Expression lowering:

            * Form and gather any required implicit expressions;
            * Apply rewrite rules;
            * Evaluate derivatives;
            * Flatten vectorial equations;
            * Indexify Functions;
            * Apply substitution rules;
            * Shift indices for domain alignment.
        """
        # Add in implicit expressions
        expressions = generate_implicit_exprs(expressions)

        # Specialization is performed on unevaluated expressions
        expressions = cls._specialize_dsl(expressions, **kwargs)

        # Lower functional DSL
        expressions = flatten([i.evaluate for i in expressions])
        expressions = [j for i in expressions for j in i._flatten]

        # A second round of specialization is performed on nevaluated expressions
        expressions = cls._specialize_exprs(expressions, **kwargs)

        # "True" lowering (indexification, shifting, ...)
        expressions = lower_exprs(expressions, **kwargs)

        processed = [LoweredEq(i) for i in expressions]

        return processed

    # Compilation -- Cluster level

    @classmethod
    def _specialize_clusters(cls, clusters, **kwargs):
        """
        Backend hook for specialization at the Cluster level.
        """
        return clusters

    @classmethod
    @timed_pass(name='lowering.Clusters')
    def _lower_clusters(cls, expressions, profiler, **kwargs):
        """
        Clusters lowering:

            * Group expressions into Clusters;
            * Introduce guards for conditional Clusters;
            * Analyze Clusters to detect computational properties such
              as parallelism.
        """
        # Build a sequence of Clusters from a sequence of Eqs
        clusters = clusterize(expressions)

        # Operation count before specialization
        init_ops = sum(estimate_cost(c.exprs) for c in clusters if c.is_dense)

        clusters = cls._specialize_clusters(clusters, **kwargs)

        # Operation count after specialization
        final_ops = sum(estimate_cost(c.exprs) for c in clusters if c.is_dense)
        profiler.record_ops_variation(init_ops, final_ops)

        return ClusterGroup(clusters)

    # Compilation -- ScheduleTree level

    @classmethod
    def _specialize_stree(cls, stree, **kwargs):
        """
        Backend hook for specialization at the Schedule tree level.
        """
        return stree

    @classmethod
    @timed_pass(name='lowering.ScheduleTree')
    def _lower_stree(cls, clusters, **kwargs):
        """
        Schedule tree lowering:

            * Turn a sequence of Clusters into a ScheduleTree;
            * Derive and attach metadata for distributed-memory parallelism;
            * Derive sections for performance profiling
        """
        # Build a ScheduleTree from a sequence of Clusters
        stree = stree_build(clusters)

        stree = cls._specialize_stree(stree)

        return stree

    # Compilation -- Iteration/Expression tree level

    @classmethod
    def _specialize_iet(cls, graph, **kwargs):
        """
        Backend hook for specialization at the Iteration/Expression tree level.
        """
        return graph

    @classmethod
    @timed_pass(name='lowering.IET')
    def _lower_iet(cls, stree, profiler, **kwargs):
        """
        Iteration/Expression tree lowering:

            * Turn a ScheduleTree into an Iteration/Expression tree;
            * Introduce distributed-memory, shared-memory, and SIMD parallelism;
            * Introduce optimizations for data locality;
            * Finalize (e.g., symbol definitions, array casts)
        """
        name = kwargs.get("name", "Kernel")
        sregistry = kwargs['sregistry']

        # Build an IET from a ScheduleTree
        iet = iet_build(stree)

        # Analyze the IET Sections for C-level profiling
        profiler.analyze(iet)

        # Wrap the IET with an EntryFunction (a special Callable representing
        # the entry point of the generated library)
        parameters = derive_parameters(iet, True)
        iet = EntryFunction(name, iet, 'int', parameters, ())

        # Lower IET to a target-specific IET
        graph = Graph(iet)
        graph = cls._specialize_iet(graph, **kwargs)

        # Instrument the IET for C-level profiling
        # Note: this is postponed until after _specialize_iet because during
        # specialization further Sections may be introduced
        instrument(graph, profiler=profiler, sregistry=sregistry)

        return graph.root, graph

    # Read-only properties exposed to the outside world

    @cached_property
    def output(self):
        return tuple(self._output)

    @cached_property
    def dimensions(self):
        return tuple(self._dimensions)

    @cached_property
    def input(self):
        ret = [i for i in self._input + list(self.parameters) if i.is_Input]
        return tuple(filter_ordered(ret))

    @cached_property
    def temporaries(self):
        return tuple(i for i in self.input if i.is_TempFunction)

    @cached_property
    def objects(self):
        return tuple(i for i in self.parameters if i.is_Object)

    # Arguments processing

    def _prepare_arguments(self, **kwargs):
        """
        Process runtime arguments passed to ``.apply()` and derive
        default values for any remaining arguments.
        """
        overrides, defaults = split(self.input, lambda p: p.name in kwargs)

        # Process data-carrier overrides
        args = ReducerMap()
        for p in overrides:
            args.update(p._arg_values(**kwargs))
            try:
                args = ReducerMap(args.reduce_all())
            except ValueError:
                raise ValueError("Override `%s` is incompatible with overrides `%s`" %
                                 (p, [i for i in overrides if i.name in args]))
        # Process data-carrier defaults
        for p in defaults:
            if p.name in args:
                # E.g., SubFunctions
                continue
            for k, v in p._arg_values(**kwargs).items():
                if k in args and args[k] != v:
                    raise ValueError("Default `%s` is incompatible with other args as "
                                     "`%s=%s`, while `%s=%s` is expected. Perhaps you "
                                     "forgot to override `%s`?" %
                                     (p, k, v, k, args[k], p))
                args[k] = v
        args = args.reduce_all()

        # All DiscreteFunctions should be defined on the same Grid
        grids = {getattr(kwargs[p.name], 'grid', None) for p in overrides}
        grids.update({getattr(p, 'grid', None) for p in defaults})
        grids.discard(None)
        if len(grids) > 1 and configuration['mpi']:
            raise ValueError("Multiple Grids found")
        try:
            grid = grids.pop()
            args.update(grid._arg_values(**kwargs))
        except KeyError:
            grid = None

        # Process Dimensions
        # A topological sorting is used so that derived Dimensions are processed after
        # their parents (note that a leaf Dimension can have an arbitrary long list of
        # ancestors)
        dag = DAG(self.dimensions,
                  [(i, i.parent) for i in self.dimensions if i.is_Derived])
        for d in reversed(dag.topological_sort()):
            args.update(d._arg_values(args, self._dspace[d], grid, **kwargs))

        # Process Objects (which may need some `args`)
        for o in self.objects:
            args.update(o._arg_values(args, grid=grid, **kwargs))

        # Sanity check
        for p in self.parameters:
            p._arg_check(args, self._dspace[p])
        for d in self.dimensions:
            if d.is_Derived:
                d._arg_check(args, self._dspace[p])

        # Turn arguments into a format suitable for the generated code
        # E.g., instead of NumPy arrays for Functions, the generated code expects
        # pointers to ctypes.Struct
        for p in self.parameters:
            try:
                args.update(kwargs.get(p.name, p)._arg_as_ctype(args, alias=p))
            except AttributeError:
                # User-provided floats/ndarray obviously do not have `_arg_as_ctype`
                args.update(p._arg_as_ctype(args, alias=p))

        # Execute autotuning and adjust arguments accordingly
        args = self._autotune(args, kwargs.pop('autotune', configuration['autotuning']))

        # Check all user-provided keywords are known to the Operator
        if not configuration['ignore-unknowns']:
            for k, v in kwargs.items():
                if k not in self._known_arguments:
                    raise ValueError("Unrecognized argument %s=%s" % (k, v))

        # Attach `grid` to the arguments map
        args = ArgumentsMap(grid, **args)

        return args

    def _postprocess_arguments(self, args, **kwargs):
        """Process runtime arguments upon returning from ``.apply()``."""
        for p in self.parameters:
            try:
                p._arg_apply(args[p.name], args[p.coordinates.name], kwargs.get(p.name))
            except AttributeError:
                p._arg_apply(args[p.name], kwargs.get(p.name))

    @cached_property
    def _known_arguments(self):
        """The arguments that can be passed to ``apply`` when running the Operator."""
        ret = set()
        for i in self.input:
            ret.update(i._arg_names)
            try:
                ret.update(i.grid._arg_names)
            except AttributeError:
                pass
        for d in self.dimensions:
            ret.update(d._arg_names)
        return frozenset(ret)

    def _autotune(self, args, setup):
        """Auto-tuning to improve runtime performance."""
        return args

    def arguments(self, **kwargs):
        """Arguments to run the Operator."""
        args = self._prepare_arguments(**kwargs)
        # Check all arguments are present
        for p in self.parameters:
            if args.get(p.name) is None:
                raise ValueError("No value found for parameter %s" % p.name)
        return args

    # JIT compilation

    @cached_property
    def _soname(self):
        """A unique name for the shared object resulting from JIT compilation."""
        return Signer._digest(self, configuration)

    def _jit_compile(self):
        """
        JIT-compile the C code generated by the Operator.

        It is ensured that JIT compilation will only be performed once per
        Operator, reagardless of how many times this method is invoked.
        """
        if self._lib is None:
            with self._profiler.timer_on('jit-compile'):
                recompiled, src_file = self._compiler.jit_compile(self._soname,
                                                                  str(self.ccode))

            elapsed = self._profiler.py_timers['jit-compile']
            if recompiled:
                perf("Operator `%s` jit-compiled `%s` in %.2f s with `%s`" %
                     (self.name, src_file, elapsed, self._compiler))
            else:
                perf("Operator `%s` fetched `%s` in %.2f s from jit-cache" %
                     (self.name, src_file, elapsed))

    @property
    def cfunction(self):
        """The JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            self._jit_compile()
            self._lib = self._compiler.load(self._soname)
            self._lib.name = self._soname

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            # Associate a C type to each argument for runtime type check
            self._cfunction.argtypes = [i._C_ctype for i in self.parameters]

        return self._cfunction

    # Execution

    def __call__(self, **kwargs):
        return self.apply(**kwargs)

    def apply(self, **kwargs):
        """
        Execute the Operator.

        With no arguments provided, the Operator runs using the data carried by the
        objects appearing in the input expressions -- these are referred to as the
        "default arguments".

        Optionally, any of the Operator default arguments may be replaced by passing
        suitable key-value arguments. Given ``apply(k=v, ...)``, ``(k, v)`` may be
        used to:

        * replace a Constant. In this case, ``k`` is the name of the Constant,
          ``v`` is either a Constant or a scalar value.

        * replace a Function (SparseFunction). Here, ``k`` is the name of the
          Function, ``v`` is either a Function or a numpy.ndarray.

        * alter the iteration interval along a Dimension. Consider a generic
          Dimension ``d`` iterated over by the Operator.  By default, the Operator
          runs over all iterations within the compact interval ``[d_m, d_M]``,
          where ``d_m`` and ``d_M`` are, respectively, the smallest and largest
          integers not causing out-of-bounds memory accesses (for the Grid
          Dimensions, this typically implies iterating over the entire physical
          domain). So now ``k`` can be either ``d_m`` or ``d_M``, while ``v``
          is an integer value.

        Examples
        --------
        Consider the following Operator

        >>> from devito import Eq, Grid, TimeFunction, Operator
        >>> grid = Grid(shape=(3, 3))
        >>> u = TimeFunction(name='u', grid=grid, save=3)
        >>> op = Operator(Eq(u.forward, u + 1))

        The Operator is run by calling ``apply``

        >>> summary = op.apply()

        The variable ``summary`` contains information about runtime performance.
        As no key-value parameters are specified, the Operator runs with its
        default arguments, namely ``u=u, x_m=0, x_M=2, y_m=0, y_M=2, time_m=0,
        time_M=1``.

        At this point, the same Operator can be used for a completely different
        run, for example

        >>> u2 = TimeFunction(name='u', grid=grid, save=5)
        >>> summary = op.apply(u=u2, x_m=1, y_M=1)

        Now, the Operator will run with a different set of arguments, namely
        ``u=u2, x_m=1, x_M=2, y_m=0, y_M=1, time_m=0, time_M=3``.

        To run an Operator that only uses buffered TimeFunctions, the maximum
        iteration point along the time dimension must be explicitly specified
        (otherwise, the Operator wouldn't know how many iterations to run).

        >>> u3 = TimeFunction(name='u', grid=grid)
        >>> op = Operator(Eq(u3.forward, u3 + 1))
        >>> summary = op.apply(time_M=10)
        """
        # Build the arguments list to invoke the kernel function
        with self._profiler.timer_on('arguments'):
            args = self.arguments(**kwargs)

        # Invoke kernel function with args
        arg_values = [args[p.name] for p in self.parameters]
        try:
            cfunction = self.cfunction
            with self._profiler.timer_on('apply', comm=args.comm):
                cfunction(*arg_values)
        except ctypes.ArgumentError as e:
            if e.args[0].startswith("argument "):
                argnum = int(e.args[0][9:].split(':')[0]) - 1
                newmsg = "error in argument '%s' with value '%s': %s" % (
                    self.parameters[argnum].name,
                    arg_values[argnum],
                    e.args[0])
                raise ctypes.ArgumentError(newmsg) from e
            else:
                raise

        # Post-process runtime arguments
        self._postprocess_arguments(args, **kwargs)

        # Output summary of performance achieved
        return self._emit_apply_profiling(args)

    # Performance profiling

    def _emit_build_profiling(self):
        if not is_log_enabled_for('PERF'):
            return

        # Rounder to K decimal places
        fround = lambda i, n=100: ceil(i * n) / n

        timings = self._profiler.py_timers.copy()

        tot = timings.pop('op-compile')
        perf("Operator `%s` generated in %.2f s" % (self.name, fround(tot)))

        max_hotspots = 3
        threshold = 20.

        def _emit_timings(timings, indent=''):
            timings.pop('total', None)
            entries = sorted(timings, key=lambda i: timings[i]['total'], reverse=True)
            for i in entries[:max_hotspots]:
                v = fround(timings[i]['total'])
                perc = fround(v/tot*100, n=10)
                if perc > threshold:
                    perf("%s%s: %.2f s (%.1f %%)" % (indent, i.lstrip('_'), v, perc))
                    _emit_timings(timings[i], ' '*len(indent) + ' * ')

        _emit_timings(timings, '  * ')

        if self._profiler._ops:
            ops = ['%d --> %d' % i for i in self._profiler._ops]
            perf("Flops reduction after symbolic optimization: [%s]" % ' ; '.join(ops))

    def _emit_apply_profiling(self, args):
        """Produce a performance summary of the profiled sections."""
        # Rounder to 2 decimal places
        fround = lambda i: ceil(i * 100) / 100

        info("Operator `%s` ran in %.2f s" % (self.name,
                                              fround(self._profiler.py_timers['apply'])))

        summary = self._profiler.summary(args, self._dtype, reduce_over='apply')

        if not is_log_enabled_for('PERF'):
            # Do not waste time
            return summary

        if summary.globals:
            # Note that with MPI enabled, the global performance indicators
            # represent "cross-rank" performance data
            metrics = []

            v = summary.globals.get('vanilla')
            if v is not None:
                metrics.append("OI=%.2f" % fround(v.oi))
                metrics.append("%.2f GFlops/s" % fround(v.gflopss))

            v = summary.globals.get('fdlike')
            if v is not None:
                metrics.append("%.2f GPts/s" % fround(v.gpointss))

            if metrics:
                perf("Global performance: [%s]" % ', '.join(metrics))

            perf("Local performance:")
            indent = " "*2
        else:
            indent = ""

        # Emit local, i.e. "per-rank" performance. Without MPI, this is the only
        # thing that will be emitted
        for k, v in summary.items():
            rank = "[rank%d]" % k.rank if k.rank is not None else ""
            oi = "OI=%.2f" % fround(v.oi)
            gflopss = "%.2f GFlops/s" % fround(v.gflopss)
            gpointss = "%.2f GPts/s" % fround(v.gpointss) if v.gpointss else None
            metrics = ", ".join(i for i in [oi, gflopss, gpointss] if i is not None)
            itershapes = [",".join(str(i) for i in its) for its in v.itershapes]
            if len(itershapes) > 1:
                itershapes = ",".join("<%s>" % i for i in itershapes)
            elif len(itershapes) == 1:
                itershapes = itershapes[0]
            else:
                itershapes = ""
            name = "%s%s<%s>" % (k.name, rank, itershapes)

            perf("%s* %s ran in %.2f s [%s]" % (indent, name, fround(v.time), metrics))
            for n, time in summary.subsections.get(k.name, {}).items():
                perf("%s+ %s ran in %.2f s [%.2f%%]" %
                     (indent*2, n, time, fround(time/v.time*100)))

        # Emit performance mode and arguments
        perf_args = {}
        for i in self.input + self.dimensions:
            if not i.is_PerfKnob:
                continue
            try:
                perf_args[i.name] = args[i.name]
            except KeyError:
                # Try with the aliases
                for a in i._arg_names:
                    if a in args:
                        perf_args[a] = args[a]
                        break
        perf("Performance[mode=%s] arguments: %s" % (self._state['optimizations'],
                                                     perf_args))

        return summary

    # Misc properties

    @cached_property
    def _mem_summary(self):
        """
        The amount of data, in bytes, used by the Operator. This is provided as
        symbolic expressions, one symbolic expression for each memory scope (external,
        heap).
        """
        roots = [self] + [i.root for i in self._func_table.values()]
        functions = [i for i in derive_parameters(roots) if i.is_Function]

        summary = {}

        external = [i.symbolic_shape for i in functions if i._mem_external]
        external = sum(reduce(mul, i, 1) for i in external)*self._dtype().itemsize
        summary['external'] = external

        heap = [i.symbolic_shape for i in functions if i._mem_heap]
        heap = sum(reduce(mul, i, 1) for i in heap)*self._dtype().itemsize
        summary['heap'] = heap

        summary['total'] = external + heap

        return summary

    # Pickling support

    def __getstate__(self):
        if self._lib:
            state = dict(self.__dict__)
            # The compiled shared-object will be pickled; upon unpickling, it
            # will be restored into a potentially different temporary directory,
            # so the entire process during which the shared-object is loaded and
            # given to ctypes must be performed again
            state['_lib'] = None
            state['_cfunction'] = None
            # Do not pickle the `args` used to construct the Operator. Not only
            # would this be completely useless, but it might also lead to
            # allocating additional memory upon unpickling, as the user-provided
            # equations typically carry different instances of the same Function
            # (e.g., f(t, x-1), f(t, x), f(t, x+1)), which are different objects
            # with distinct `.data` fields
            state['_args'] = None
            with open(self._lib._name, 'rb') as f:
                state['binary'] = f.read()
                state['soname'] = self._soname
            return state
        else:
            return self.__dict__

    def __getnewargs_ex__(self):
        return (None,), {}

    def __setstate__(self, state):
        soname = state.pop('soname', None)
        binary = state.pop('binary', None)
        for k, v in state.items():
            setattr(self, k, v)
        if soname is not None:
            self._compiler.save(soname, binary)
            self._lib = self._compiler.load(soname)
            self._lib.name = soname


# Misc helpers


class ArgumentsMap(dict):

    def __init__(self, grid, *args, **kwargs):
        super(ArgumentsMap, self).__init__(*args, **kwargs)
        self.grid = grid

    @property
    def comm(self):
        """The MPI communicator the arguments are collective over."""
        return self.grid.comm if self.grid is not None else MPI.COMM_NULL


def parse_kwargs(**kwargs):
    """
    Parse keyword arguments provided to an Operator.
    """
    # `dse` -- deprecated, dropped
    dse = kwargs.pop("dse", None)
    if dse is not None:
        warning("The `dse` argument is deprecated. "
                "The optimization level is now controlled via the `opt` argument")

    # `dle` -- deprecated, replaced by `opt`
    if 'dle' in kwargs:
        warning("The `dle` argument is deprecated. "
                "The optimization level is now controlled via the `opt` argument")
        dle = kwargs.pop('dle')
        if 'opt' in kwargs:
            warning("Both `dle` and `opt` were passed; ignoring `dle` argument")
            opt = kwargs.pop('opt')
        else:
            warning("Setting `opt=%s`" % str(dle))
            opt = dle
    elif 'opt' in kwargs:
        opt = kwargs.pop('opt')
    else:
        opt = configuration['opt']

    if not opt or isinstance(opt, str):
        mode, options = opt, {}
    elif isinstance(opt, tuple):
        if len(opt) == 0:
            mode, options = 'noop', {}
        elif isinstance(opt[-1], dict):
            if len(opt) == 2:
                mode, options = opt
            else:
                mode, options = tuple(flatten(i.split(',') for i in opt[:-1])), opt[-1]
        else:
            mode, options = tuple(flatten(i.split(',') for i in opt)), {}
    else:
        raise InvalidOperator("Illegal `opt=%s`" % str(opt))

    # `opt`, deprecated kwargs
    kwopenmp = kwargs.get('openmp', options.get('openmp'))
    if kwopenmp is None:
        openmp = kwargs.get('language', configuration['language']) == 'openmp'
    else:
        openmp = kwopenmp

    # `opt`, options
    options = dict(options)
    options.setdefault('openmp', openmp)
    options.setdefault('mpi', configuration['mpi'])
    for k, v in configuration['opt-options'].items():
        options.setdefault(k, v)
    kwargs['options'] = options

    # `opt`, mode
    if mode is None:
        mode = 'noop'
    kwargs['mode'] = mode

    # `platform`
    platform = kwargs.get('platform')
    if platform is not None:
        if not isinstance(platform, str):
            raise ValueError("Argument `platform` should be a `str`")
        if platform not in configuration._accepted['platform']:
            raise InvalidOperator("Illegal `platform=%s`" % str(platform))
        kwargs['platform'] = platform_registry[platform]()
    else:
        kwargs['platform'] = configuration['platform']

    # `language`
    language = kwargs.get('language')
    if language is not None:
        if not isinstance(language, str):
            raise ValueError("Argument `language` should be a `str`")
        if language not in configuration._accepted['language']:
            raise InvalidOperator("Illegal `language=%s`" % str(language))
        kwargs['language'] = language
    elif kwopenmp is not None:
        # Handle deprecated `openmp` kwarg for backward compatibility
        kwargs['language'] = 'openmp' if openmp else 'C'
    else:
        kwargs['language'] = configuration['language']

    # `compiler`
    compiler = kwargs.get('compiler')
    if compiler is not None:
        if not isinstance(compiler, str):
            raise ValueError("Argument `compiler` should be a `str`")
        if compiler not in configuration._accepted['compiler']:
            raise InvalidOperator("Illegal `compiler=%s`" % str(compiler))
        kwargs['compiler'] = compiler_registry[compiler](platform=kwargs['platform'],
                                                         language=kwargs['language'])
    elif any([platform, language]):
        kwargs['compiler'] =\
            configuration['compiler'].__new_from__(platform=kwargs['platform'],
                                                   language=kwargs['language'])
    else:
        kwargs['compiler'] = configuration['compiler']

    return kwargs
