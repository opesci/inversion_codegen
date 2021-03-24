from collections import defaultdict
from itertools import groupby

from devito.ir.support import Scope
from devito.tools import as_tuple, flatten

__all__ = ['Queue', 'QueueStateful']


class Queue(object):

    """
    A special queue to process Clusters based on a divide-and-conquer algorithm.

    Notes
    -----
    Subclasses must override :meth:`callback`, which may get executed either
    before (fdta -- first divide then apply) or after (fatd -- first apply
    then divide) the divide phase of the algorithm.
    """

    def callback(self, *args):
        raise NotImplementedError

    def process(self, clusters):
        return self._process_fdta(clusters, 1)

    def _make_key(self, element, level):
        itintervals = element.itintervals[:level]

        subkey = self._make_key_hook(element, level)

        return (itintervals,) + subkey

    def _make_key_hook(self, element, level):
        return ()

    def _process_fdta(self, clusters, level, prefix=None, **kwargs):
        """
        fdta -> First Divide Then Apply
        """
        prefix = prefix or []

        # Divide part
        processed = []
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Recursion
                processed.extend(self._process_fdta(list(g), level + 1, pfx, **kwargs))

        # Apply callback
        processed = self.callback(processed, prefix, **kwargs)

        return processed

    def _process_fatd(self, clusters, level, **kwargs):
        """
        fatd -> First Apply Then Divide
        """
        # Divide part
        processed = []
        for k, g in groupby(clusters, key=lambda i: self._make_key(i, level)):
            pfx = k[0]
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Apply callback
                _clusters = self.callback(list(g), pfx, **kwargs)
                # Recursion
                processed.extend(self._process_fatd(_clusters, level + 1, **kwargs))

        return processed


class QueueStateful(Queue):

    """
    A Queue carrying along some state. This is useful when one wants to avoid
    expensive re-computations of information.
    """

    class State(object):

        def __init__(self):
            self.properties = {}
            self.scopes = {}

    def __init__(self, state=None):
        super(QueueStateful, self).__init__()
        self.state = state or QueueStateful.State()

    def _fetch_scope(self, clusters):
        exprs = flatten(c.exprs for c in as_tuple(clusters))
        key = tuple(exprs)
        if key not in self.state.scopes:
            self.state.scopes[key] = Scope(exprs)
        return self.state.scopes[key]

    def _fetch_properties(self, clusters, prefix):
        # If the situation is:
        #
        # t
        #   x0
        #     <some clusters>
        #   x1
        #     <some other clusters>
        #
        # then retain only the "common" properties, that is those along `t`
        properties = defaultdict(set)
        for c in clusters:
            v = self.state.properties.get(c, {})
            for i in prefix:
                properties[i.dim].update(v.get(i.dim, set()))
        return properties
