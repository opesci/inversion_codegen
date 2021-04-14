from collections import OrderedDict
from itertools import combinations, product
from functools import total_ordering

from devito.arch import KNL, KNL7210
from devito.ir import Backward, retrieve_iteration_tree
from devito.logger import perf, warning as _warning
from devito.mpi.distributed import MPI, MPINeighborhood
from devito.mpi.routines import MPIMsgEnriched
from devito.parameters import configuration
from devito.tools import filter_ordered, flatten, is_integer, prod
from devito.types import Timer

__all__ = ['autotune']


def autotune(operator, args, level, mode):
    """
    Operator autotuning.

    Parameters
    ----------
    operator : Operator
        Input Operator.
    args : dict_like
        The runtime arguments with which `operator` is run.
    level : str
        The autotuning aggressiveness (basic, aggressive, max). A more
        aggressive autotuning might eventually result in higher runtime
        performance, but the autotuning phase will take longer.
    mode : str
        The autotuning mode (preemptive, runtime). In preemptive mode, the
        output runtime values supplied by the user to `operator.apply` are
        replaced with shadow copies.
    """
    key = [level, mode]
    accepted = configuration._accepted['autotuning']
    if key not in accepted:
        raise ValueError("The accepted `(level, mode)` combinations are `%s`; "
                         "provided `%s` instead" % (accepted, key))

    # We get passed all the arguments, but the cfunction only requires a subset
    at_args = OrderedDict([(p.name, args[p.name]) for p in operator.parameters])

    # User-provided output data won't be altered in `preemptive` mode
    if mode == 'preemptive':
        output = {i.name: i for i in operator.output}
        copies = {k: output[k]._C_as_ndarray(v).copy()
                  for k, v in args.items() if k in output}
        # WARNING: `copies` keeps references to numpy arrays, which is required
        # to avoid garbage collection to kick in during autotuning and prematurely
        # free the shadow copies handed over to C-land
        at_args.update({k: output[k]._C_make_dataobj(v) for k, v in copies.items()})

    # Disable halo exchanges through MPI_PROC_NULL
    if mode in ['preemptive', 'destructive']:
        for p in operator.parameters:
            if isinstance(p, MPINeighborhood):
                at_args.update(MPINeighborhood(p.neighborhood)._arg_values())
                for i in p.fields:
                    setattr(at_args[p.name]._obj, i, MPI.PROC_NULL)
            elif isinstance(p, MPIMsgEnriched):
                at_args.update(MPIMsgEnriched(p.name, p.function, p.halos)._arg_values())
                for i in at_args[p.name]:
                    i.fromrank = MPI.PROC_NULL
                    i.torank = MPI.PROC_NULL

    roots = [operator.body] + [i.root for i in operator._func_table.values()]
    trees = filter_ordered(retrieve_iteration_tree(roots), key=lambda i: i.root)

    # Detect the time-stepping Iteration; shrink its iteration range so that
    # each autotuning run only takes a few iterations
    steppers = {i for i in flatten(trees) if i.dim.is_Time}
    if len(steppers) == 0:
        stepper = None
        timesteps = 1
    elif len(steppers) == 1:
        stepper = steppers.pop()
        timesteps = init_time_bounds(stepper, at_args, args)
        if timesteps is None:
            return args, {}
    else:
        warning("cannot perform autotuning unless there is one time loop; skipping")
        return args, {}

    # Use a fresh Timer for auto-tuning
    timer = Timer('timers', list(operator._profiler.all_sections))
    at_args.update(timer._arg_values())

    # Perform autotuning
    # import pdb;pdb.set_trace()
    timings = {}
    for n, tree in enumerate(trees):
        blockable = [i.dim for i in tree if not is_integer(i.step)]

        # Tunable arguments
        try:
            tunable = []
            tunable.append(generate_block_shapes(blockable, args, level))
            tunable.append(generate_nthreads(operator.nthreads, args, level))
            tunable = list(product(*tunable))
        except ValueError:
            # Some arguments are compulsory, otherwise autotuning is skipped
            continue

        # Symbolic number of loop-blocking blocks per thread
        nblocks_per_thread = calculate_nblocks(tree, blockable) / operator.nthreads

        for bs, nt in tunable:
            # Can we safely autotune over the given time range?
            if not check_time_bounds(stepper, at_args, args, mode):
                break

            # Update `at_args` to use the new tunable arguments
            run = [(k, v) for k, v in bs + nt if k in at_args]
            at_args.update(dict(run))

            # Drop run if not at least one block per thread
            if not configuration['develop-mode'] and nblocks_per_thread.subs(at_args) < 1:
                continue

            # Run the Operator
            operator.cfunction(*list(at_args.values()))

            # Record timing
            elapsed = timer.total
            timings.setdefault(nt, OrderedDict()).setdefault(n, {})[bs] = elapsed
            log("run <%s> took %f (s) in %d timesteps" %
                (','.join('%s=%s' % i for i in run), elapsed, timesteps))

            # Prepare for the next autotuning run
            update_time_bounds(stepper, at_args, timesteps, mode)
            timer.reset()

    # The best variant is the one that for a given number of threads had the minium
    # turnaround time
    try:
        runs = 0
        mapper = {}
        for k, v in timings.items():
            for i in v.values():
                runs += len(i)
                record = mapper.setdefault(k, Record())
                record.add(min(i, key=i.get), min(i.values()))
        best = min(mapper, key=mapper.get)
        best = OrderedDict(best + tuple(mapper[best].args))
        best.pop(None, None)
        log("selected <%s>" % (','.join('%s=%s' % i for i in best.items())))
    except ValueError:
        warning("could not perform any runs")
        return args, {}

    # Update the argument list with the tuned arguments
    args.update(best)

    # In `runtime` mode, some timesteps have been executed already, so we must
    # adjust the time range
    finalize_time_bounds(stepper, at_args, args, mode)

    # Autotuning summary
    summary = {}
    summary['runs'] = runs
    summary['tpr'] = timesteps  # tpr -> timesteps per run
    summary['tuned'] = dict(best)

    return args, summary


@total_ordering
class Record(object):

    def __init__(self):
        self.args = []
        self.time = 0

    def __repr__(self):
        return str((self.args, self.time))

    def add(self, args, time):
        self.args.extend(list(args))
        self.time += time

    def __eq__(self, other):
        return self.time == other.time

    def __lt__(self, other):
        return self.time < other.time


def init_time_bounds(stepper, at_args, args):
    if stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        at_args[dim.min_name] = at_args[dim.max_name] - options['squeezer']
        if dim.size_name in args:
            # May need to shrink to avoid OOB accesses
            at_args[dim.min_name] = max(at_args[dim.min_name], args[dim.min_name])
        if at_args[dim.max_name] < at_args[dim.min_name]:
            warning("too few time iterations; skipping")
            return False
    else:
        at_args[dim.max_name] = at_args[dim.min_name] + options['squeezer']
        if dim.size_name in args:
            # May need to shrink to avoid OOB accesses
            at_args[dim.max_name] = min(at_args[dim.max_name], args[dim.max_name])
        if at_args[dim.min_name] > at_args[dim.max_name]:
            warning("too few time iterations; skipping")
            return False

    return stepper.size(at_args[dim.min_name], at_args[dim.max_name])


def check_time_bounds(stepper, at_args, args, mode):
    if mode != 'runtime' or stepper is None:
        return True
    dim = stepper.dim.root
    if stepper.direction is Backward:
        if at_args[dim.min_name] < args[dim.min_name]:
            warning("too few time iterations; stopping")
            return False
    else:
        if at_args[dim.max_name] > args[dim.max_name]:
            warning("too few time iterations; stopping")
            return False
    return True


def update_time_bounds(stepper, at_args, timesteps, mode):
    if mode != 'runtime' or stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        at_args[dim.max_name] -= timesteps
        at_args[dim.min_name] -= timesteps
    else:
        at_args[dim.min_name] += timesteps
        at_args[dim.max_name] += timesteps


def finalize_time_bounds(stepper, at_args, args, mode):
    if mode != 'runtime' or stepper is None:
        return
    dim = stepper.dim.root
    if stepper.direction is Backward:
        args[dim.max_name] = at_args[dim.max_name]
        args[dim.min_name] = args[dim.min_name]
    else:
        args[dim.min_name] = at_args[dim.min_name]
        args[dim.max_name] = args[dim.max_name]


def calculate_nblocks(tree, blockable):
    collapsed = tree[:(tree[0].ncollapsed or 1)]
    blocked = [i.dim for i in collapsed if i.dim in blockable]
    remainders = [(d.root.symbolic_max-d.root.symbolic_min+1) % d.step for d in blocked]
    niters = [d.root.symbolic_max - i for d, i in zip(blocked, remainders)]
    nblocks = prod((i - d.root.symbolic_min + 1) / d.step
                   for d, i in zip(blocked, niters))
    return nblocks


def generate_block_shapes(blockable, args, level):
    if not blockable:
        raise ValueError

    mapper = OrderedDict()
    for d in blockable:
        mapper[d] = mapper.get(d.parent, -1) + 1

    # Generate level-0 block shapes
    level_0 = [d for d, v in mapper.items() if v == 0]
    # Max attemptable block shape
    max_bs = tuple((d.step, d.symbolic_size.subs(args)) for d in level_0)
    # Defaults (basic mode)
    ret = [tuple((d.step, v) for d in level_0) for v in options['blocksize-l0']]
    # Always try the entire iteration space (degenerate block)
    ret.append(max_bs)
    # More attempts if autotuning in aggressive mode
    if level in ['aggressive', 'max']:
        # Ramp up to larger block shapes
        handle = tuple((i, options['blocksize-l0'][-1]) for i, _ in ret[0])
        for i in range(3):
            new_bs = tuple((b, v*2) for b, v in handle)
            ret.insert(ret.index(handle) + 1, new_bs)
            handle = new_bs
        handle = []
        # Extended shuffling for the smaller block shapes
        for bs in ret[:4]:
            for i in ret:
                handle.append(bs[:-1] + (i[-1],))
        # Some more shuffling for all block shapes
        for bs in list(ret):
            ncombs = len(bs)
            for i in range(ncombs):
                for j in combinations(dict(bs), i+1):
                    handle.append(tuple((b, v*2 if b in j else v) for b, v in bs))
        ret.extend(handle)
    # Drop block shapes exceeding the iteration space extent
    ret = [i for i in ret if all(dict(i)[k] <= v for k, v in max_bs)]
    # Drop redundant block shapes
    ret = filter_ordered(ret)

    # Generate level-1 block shapes
    level_1 = [d for d, v in mapper.items() if v == 1]
    if level_1:
        assert len(level_1) == len(level_0)
        assert all(d1.parent is d0 for d0, d1 in zip(level_0, level_1))
        for bs in list(ret):
            handle = []
            for v in options['blocksize-l1']:
                # To be a valid blocksize, it must be smaller than and divide evenly
                # the parent's block size
                if all(v <= i and i % v == 0 for _, i in bs):
                    ret.append(bs + tuple((d.step, v) for d in level_1))
            ret.remove(bs)

    # Generate level-n (n > 1) block shapes
    # TODO -- currently, there's no Operator producing depth>2 hierarchical blocking,
    # so for simplicity we ignore this for the time being

    # Normalize
    ret = [tuple((k.name, v) for k, v in bs) for bs in ret]

    return ret


def generate_nthreads(nthreads, args, level):
    if nthreads == 1:
        return [((None, 1),)]

    ret = []
    basic = ((nthreads.name, args[nthreads.name]),)

    if level != 'max':
        ret.append(basic)
    else:
        # Be sure to try with:
        # 1) num_threads == num_physical_cores
        # 2) num_threads == num_logical_cores
        platform = configuration['platform']
        name = nthreads.name
        if platform in (KNL, KNL7210):
            cases = filter_ordered([platform.cores_physical,
                                    platform.cores_physical * 2,
                                    platform.cores_logical])
        else:
            cases = filter_ordered([platform.cores_physical,
                                    platform.cores_logical])
        ret.extend([((name, nthread),) for nthread in cases])

        if basic not in ret:
            warning("skipping `%s`; perhaps you've set OMP_NUM_THREADS to a "
                    "non-standard value while attempting autotuning in "
                    "`max` mode?" % dict(basic))

    return ret


options = {
    'squeezer': 4,
    'blocksize-l0': (8, 16, 24, 32, 64, 96, 128),
    'blocksize-l1': (8, 16, 32),
}
"""Autotuning options."""


def log(msg):
    perf("AutoTuner: %s" % msg)


def warning(msg):
    _warning("AutoTuner: %s" % msg)
