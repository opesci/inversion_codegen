__all__ = ['Evaluable']


class Evaluable(object):

    """
    A mixin class for types inherited from SymPy that may carry nested
    unevaluated arguments.

    This mixin is used to implement lazy evaluation of expressions.
    """

    @classmethod
    def _evaluate_maybe_nested(cls, maybe_evaluable):
        if isinstance(maybe_evaluable, Evaluable):
            return maybe_evaluable.evaluate
        try:
            # Not an Evaluable, but some Evaluables may still be hidden within `args`
            if maybe_evaluable.args:
                args = [Evaluable._evaluate_maybe_nested(i) for i in maybe_evaluable.args]
                evaluate = not all(i is j for i, j in zip(args, maybe_evaluable.args))
                try:
                    return maybe_evaluable.func(*args, evaluate=evaluate)
                except TypeError:
                    # Not all objects support the `evaluate` kwarg
                    return maybe_evaluable.func(*args)
            else:
                return maybe_evaluable
        except AttributeError:
            # No `args` to be visited
            return maybe_evaluable

    @property
    def args(self):
        return ()

    @property
    def func(self):
        return self.__class__

    def _evaluate_args(self):
        return [Evaluable._evaluate_maybe_nested(i) for i in self.args]

    @property
    def evaluate(self):
        """Return a new object from the evaluation of ``self``."""
        args = self._evaluate_args()
        evaluate = not all(i is j for i, j in zip(args, self.args))
        return self.func(*args, evaluate=evaluate)
