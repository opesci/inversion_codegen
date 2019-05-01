from devito.logger import warning
from devito.ir.iet import find_affine_trees
from devito.operator import Operator
from devito.ops.transformer import opsit

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    def _specialize_iet(self, iet, **kwargs):
        for n, (section, trees) in enumerate(find_affine_trees(iet).items()):
            callable_kernel, par_loop_call_block = opsit(trees, n)

            self._callables.append(callable_kernel)

        warning("The OPS backend is still work-in-progress")

        return iet
