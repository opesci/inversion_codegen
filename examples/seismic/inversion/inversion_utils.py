import numpy as np

from devito import Operator, Eq, Min, Max


def compute_residual(res, dobs, dsyn):
    """
    Computes the data residual dsyn - dobs into residual
    """
    if res.grid.distributor.is_parallel:
        # If we run with MPI, we have to compute the residual via an operator
        # First make sure we can take the difference and that receivers are at the
        # same position
        assert np.allclose(dobs.coordinates.data[:], dsyn.coordinates.data)
        assert np.allclose(res.coordinates.data[:], dsyn.coordinates.data)
        # Create a difference operator
        diff_eq = Eq(res, dsyn.subs({dsyn.dimensions[-1]: res.dimensions[-1]}) -
                     dobs.subs({dobs.dimensions[-1]: res.dimensions[-1]}))
        Operator(diff_eq)()
    else:
        # A simple data difference is enough in serial
        res.data[:] = dsyn.data[:] - dobs.data[:]

    return res


def update_with_box(vp, alpha, dm, vmin=2.0, vmax=3.5):
    """
    Apply gradient update in-place to vp with box constraint

    Notes:
    ------
    For more advanced algorithm, one will need to gather the non-distributed
    velocity array to apply constrains and such.
    """
    update = vp + alpha * dm
    update_eq = Eq(vp, Max(Min(update, vmax), vmin))
    Operator(update_eq)()
