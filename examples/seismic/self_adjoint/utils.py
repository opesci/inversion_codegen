import numpy as np
from devito import Eq, Operator, SubDimension, exp, Min, Abs

__all__ = ['setup_w_over_q']


def setup_w_over_q(wOverQ, w, qmin, qmax, npad, sigma=0):
    """
    Initialise spatially variable w/Q field used to implement attenuation and
    absorb outgoing waves at the edges of the model. Uses Devito Operator.

    Parameters
    ----------
    wOverQ : Function, required
        The omega over Q field used to implement attenuation in the model,
        and the absorbing boundary condition for outgoing waves.
    w : float32, required
        center angular frequency, e.g. peak frequency of Ricker source wavelet
        used for modeling.
    qmin : float32, required
        Q value at the edge of the model. Typically set to 0.1 to strongly
        attenuate outgoing waves.
    qmax : float32, required
        Q value in the interior of the model. Typically set to 100 as a
        reasonable and physically meaningful Q value.
    npad : int, required
        Number of points in the absorbing boundary region. Note that we expect
        this to be the same on all sides of the model.
    sigma : float32, optional, defaults to None
        sigma value for call to scipy gaussian smoother, default 5.
    """
    # sanity checks
    assert w > 0, "supplied w value [%f] must be positive" % (w)
    assert qmin > 0, "supplied qmin value [%f] must be positive" % (qmin)
    assert qmax > 0, "supplied qmax value [%f] must be positive" % (qmax)
    assert npad > 0, "supplied npad value [%f] must be positive" % (npad)
    for n in wOverQ.grid.shape:
        if n - 2*npad < 1:
            raise ValueError("2 * npad must not exceed dimension size!")

    lqmin = np.log(qmin)
    lqmax = np.log(qmax)

    # 1. Get distance to closest boundary in all dimensions
    # 2. Logarithmic variation between qmin, qmax across the absorbing boundary
    eqs = [Eq(wOverQ, 1)]
    for d in wOverQ.dimensions:
        # left
        dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                  thickness=npad)
        pos = Abs(dim_l - d.symbolic_min) / float(npad)
        eqs.append(Eq(wOverQ.subs({d: dim_l}), Min(wOverQ.subs({d: dim_l}), pos)))
        # right
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=npad)
        pos = Abs(d.symbolic_max - dim_r) / float(npad)
        eqs.append(Eq(wOverQ.subs({d: dim_r}), Min(wOverQ.subs({d: dim_r}), pos)))

    eqs.append(Eq(wOverQ, w / exp(lqmin + wOverQ * (lqmax - lqmin))))
    # 2020.05.04 currently does not support spatial smoothing of the Q field
    # due to MPI weirdness in reassignment of the numpy array
    Operator(eqs, name='WOverQ_Operator')()
