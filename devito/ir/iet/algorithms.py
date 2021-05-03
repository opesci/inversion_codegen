from collections import OrderedDict

from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle, Break)
from devito.tools import timed_pass

__all__ = ['iet_build']


@timed_pass(name='build')
def iet_build(stree):
    """
    Construct an Iteration/Expression tree(IET) from a ScheduleTree.
    """
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            return List(body=queues.pop(i))

        elif i.is_Exprs:
            exprs = [Increment(e) if e.is_Increment else Expression(e) for e in i.exprs]
            body = ExpressionBundle(i.ispace, i.ops, i.traffic, body=exprs)

        elif i.is_Conditional:
            e = queues.pop(i)
            body = Conditional(i.guard.condition, e, Break() if i.guard.brk else None)

        elif i.is_Iteration:
            body = Iteration(queues.pop(i), i.dim, i.limits, direction=i.direction,
                             properties=i.properties, uindices=i.sub_iterators)

        elif i.is_Section:
            body = Section('section%d' % nsections, body=queues.pop(i))
            nsections += 1

        elif i.is_Halo:
            body = HaloSpot(i.halo_scheme, body=queues.pop(i))

        elif i.is_Sync:
            body = SyncSpot(i.sync_ops, body=queues.pop(i))

        queues.setdefault(i.parent, []).append(body)

    assert False
