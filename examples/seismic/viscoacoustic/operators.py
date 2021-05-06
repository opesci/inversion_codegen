import sympy as sp
import numpy as np

from devito import (Eq, Operator, VectorTimeFunction, TimeFunction, Function, NODE,
                    div, grad)
from examples.seismic import PointSource, Receiver


def src_rec(p, model, geometry, **kwargs):
    """
    Forward case: Source injection and receiver interpolation
    Adjoint case: Receiver injection and source interpolation
    """
    dt = model.grid.time_dim.spacing
    m = model.m
    # Source symbol with input wavelet
    src = PointSource(name="src", grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    forward = kwargs.get('forward', True)
    time_order = p.time_order

    scale = dt / m if time_order == 1 else dt**2 / m

    if forward:
        # The source injection term
        src_term = src.inject(field=p.forward, expr=src * scale)
        # Create interpolation expression for receivers
        rec_term = rec.interpolate(expr=p)
    else:
        # Construct expression to inject receiver values
        rec_term = rec.inject(field=p.backward, expr=rec * scale)
        # Create interpolation expression for the adjoint-source
        src_term = src.interpolate(expr=p)

    return [src_term, rec_term]


def sls_1st_order(model, geometry, p, r=None, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    from Blanch and Symes (1995) / Dutta and Schuster (2014).

    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    r : TimeFunction
        Memory variable.
    """
    forward = kwargs.get('forward', True)
    space_order = p.space_order
    save = kwargs.get('save', False)
    save_t = geometry.nt if save else None
    s = model.grid.stepping_dim.spacing
    b = model.b
    vp = model.vp
    damp = model.damp
    qp = model.qp
    f0 = geometry._f0
    q = kwargs.get('q', 0)

    # Particle Velocity
    v = kwargs.pop('v')

    # The stress relaxation parameter
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0

    # The strain relaxation parameter
    t_ep = 1./(f0**2*t_s)

    # The relaxation time
    tt = (t_ep/t_s)-1.

    # Density
    rho = 1. / b

    # Bulk modulus
    bm = rho * vp**2

    # Attenuation Memory variable.
    r = r or TimeFunction(name="r", grid=model.grid, time_order=1,
                          space_order=space_order, save=save_t, staggered=NODE)

    if forward:

        # Define PDE
        pde_v = v - s * b * grad(p)
        u_v = Eq(v.forward, damp * pde_v)

        pde_r = r - s * (1. / t_s) * r - s * (1. / t_s) * tt * rho * div(v.forward)
        u_r = Eq(r.forward, damp * pde_r)

        pde_p = p - s * bm * (tt + 1.) * div(v.forward) - s * vp**2 * r.forward + \
            s * vp**2 * q
        u_p = Eq(p.forward, damp * pde_p)

        return [u_v, u_r, u_p]

    else:

        # Define PDE
        pde_r = r - s * (1. / t_s) * r - s * p
        u_r = Eq(r.backward, damp * pde_r)

        pde_v = v + s * grad(rho * (1. + tt) * p) + s * \
            grad((1. / t_s) * rho * tt * r.backward)
        u_v = Eq(v.backward, damp * pde_v)

        pde_p = p + s * vp**2 * div(b * v.backward)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_r, u_v, u_p]


def sls_2nd_order(model, geometry, p, r=None, **kwargs):
    """
    Implementation of the 2nd order viscoacoustic wave-equation from Bai (2014).

    https://library.seg.org/doi/10.1190/geo2013-0030.1

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    r : TimeFunction
        Attenuation Memory variable.
    """
    forward = kwargs.get('forward', True)
    space_order = p.space_order
    save = kwargs.get('save', False)
    save_t = geometry.nt if save else None
    s = model.grid.stepping_dim.spacing
    b = model.b
    vp = model.vp
    damp = model.damp
    qp = model.qp
    f0 = geometry._f0
    q = kwargs.get('q', 0)

    # The stress relaxation parameter
    t_s = (sp.sqrt(1.+1./qp**2)-1./qp)/f0

    # The strain relaxation parameter
    t_ep = 1./(f0**2*t_s)

    # The relaxation time
    tt = (t_ep/t_s)-1.

    # Density
    rho = 1. / b

    # Bulk modulus
    bm = rho * vp**2

    # Attenuation Memory variable.
    r = r or TimeFunction(name="r", grid=model.grid, time_order=2,
                          space_order=space_order, save=save_t, staggered=NODE)

    if forward:

        pde_r = r + s * (tt / t_s) * rho * div(b * grad(p, shift=.5), shift=-.5) - \
            s * (1. / t_s) * r
        u_r = Eq(r.forward, damp * pde_r)

        pde_p = 2. * p - damp * p.backward + s**2 * bm * (1. + tt) * \
            div(b * grad(p, shift=.5), shift=-.5) - s**2 * vp**2 * \
            r.forward + s**2 * vp**2 * q
        u_p = Eq(p.forward, damp * pde_p)

        return [u_r, u_p]

    else:

        pde_r = r + s * (tt / t_s) * p - s * (1. / t_s) * r
        u_r = Eq(r.backward, damp * pde_r)

        pde_p = 2. * p - damp * p.forward + s**2 * vp**2 * \
            div(b * grad((1. + tt) * rho * p, shift=.5), shift=-.5) - s**2 * vp**2 * \
            div(b * grad(rho * r.backward, shift=.5), shift=-.5)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_r, u_p]


def ren_1st_order(model, geometry, p, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation from Ren et al. (2014).

    https://academic.oup.com/gji/article/197/2/948/616510

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)
    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Particle velocity
    v = kwargs.pop('v')

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    eta = vp**2 / (w0 * qp)

    # Bulk modulus
    bm = rho * vp**2

    if forward:

        # Define PDE
        pde_v = v - s * b * grad(p)
        u_v = Eq(v.forward, damp * pde_v)

        pde_p = p - s * bm * div(v.forward) + \
            s * eta * rho * div(b * grad(p, shift=.5), shift=-.5)
        u_p = Eq(p.forward, damp * pde_p)

        return [u_v, u_p]

    else:

        pde_v = v + s * grad(bm * p)
        u_v = Eq(v.backward, pde_v * damp)

        pde_p = p + s * div(b * grad(rho * eta * p, shift=.5), shift=-.5) + \
            s * div(b * v.backward)
        u_p = Eq(p.backward, pde_p * damp)

        return [u_v, u_p]


def ren_2nd_order(model, geometry, p, **kwargs):
    """
    Implementation of the 2nd order viscoacoustic wave-equation from Ren et al. (2014).

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)

    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    eta = vp**2 / (w0 * qp)

    # Bulk modulus
    bm = rho * vp**2

    if forward:

        pde_p = 2. * p - damp * p.backward + s**2 * bm * \
            div(b * grad(p, shift=.5), shift=-.5) + s**2 * eta * rho * \
            div(b * grad(p - p.backward, shift=.5) / s, shift=-.5)

        u_p = Eq(p.forward, damp * pde_p)

        return [u_p]

    else:

        pde_p = 2. * p - damp * p.forward + s**2 * \
            div(b * grad(bm * p, shift=.5), shift=-.5) - s**2 * \
            div(b * grad(((p.forward - p) / s) * rho * eta, shift=.5), shift=-.5)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_p]


def deng_1st_order(model, geometry, p, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    from Deng and McMechan (2007).

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)
    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Particle velocity
    v = kwargs.pop('v')

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    # Bulk modulus
    bm = rho * vp**2

    if forward:

        # Define PDE
        pde_v = v - s * b * grad(p)
        u_v = Eq(v.forward, damp * pde_v)

        pde_p = p - s * bm * div(v.forward) - s * (w0 / qp) * p
        u_p = Eq(p.forward, damp * pde_p)

        return [u_v, u_p]

    else:

        pde_v = v + s * grad(bm * p)
        u_v = Eq(v.backward, pde_v * damp)

        pde_p = p + s * div(b * v.backward) - s * (w0 / qp) * p
        u_p = Eq(p.backward, pde_p * damp)

        return [u_v, u_p]


def deng_2nd_order(model, geometry, p, **kwargs):
    """
    Implementation of the 2nd order viscoacoustic wave-equation
    from Deng and McMechan (2007).

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    forward = kwargs.get('forward', True)

    s = model.grid.stepping_dim.spacing
    f0 = geometry._f0
    vp = model.vp
    b = model.b
    qp = model.qp
    damp = model.damp

    # Angular frequency
    w0 = 2. * np.pi * f0

    # Density
    rho = 1. / b

    bm = rho * vp**2

    if forward:

        pde_p = 2. * p - damp*p.backward + s**2 * bm * \
            div(b * grad(p, shift=.5), shift=-.5) - s**2 * w0/qp * (p - p.backward)/s
        u_p = Eq(p.forward, damp * pde_p)

        return [u_p]

    else:

        pde_p = 2. * p - damp * p.forward + s**2 * w0 / qp * (p.forward - p) / s + \
            s * s * div(b * grad(bm * p, shift=.5), shift=-.5)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_p]


def sls(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation
    from Blanch and Symes (1995) / Dutta and Schuster (2014) and
    Implementation of the 2nd order viscoacoustic wave-equation from Bai (2014).

    https://library.seg.org/doi/pdf/10.1190/1.1822695
    https://library.seg.org/doi/pdf/10.1190/geo2013-0414.1
    https://library.seg.org/doi/10.1190/geo2013-0030.1

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('sls', time_order)]
    eqn = eq_stencil(model, geometry, p, forward=forward, **kwargs)

    return eqn


def ren(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the 1st and 2nd order viscoacoustic wave-equation from
    Ren et al. (2014).

    https://academic.oup.com/gji/article/197/2/948/616510
    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('ren', time_order)]
    eqn = eq_stencil(model, geometry, p, forward=forward, **kwargs)

    return eqn


def deng_mcmechan(model, geometry, p, forward=True, **kwargs):
    """
    Implementation of the 1st order viscoacoustic wave-equation and 2nd order
    viscoacoustic wave-equation from Deng and McMechan (2007).

    https://library.seg.org/doi/pdf/10.1190/1.2714334

    Parameters
    ----------
    p : TimeFunction
        Pressure field.
    """
    time_order = p.time_order

    eq_stencil = stencils[('deng_mcmechan', time_order)]
    eqn = eq_stencil(model, geometry, p, forward=forward, **kwargs)

    return eqn


def ForwardOperator(model, geometry, space_order=4, kernel='sls', time_order=2,
                    save=False, **kwargs):
    """
    Construct method for the forward modelling operator in a viscoacoustic medium.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : string, optional
        selects a viscoacoustic equation from the options below:
        sls (Standard Linear Solid) :
        1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        2nd order - Bai et al. (2014) viscoacoustic equation
        ren - Ren et al. (2014) viscoacoustic equation
        deng_mcmechan - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """
    # Create symbols for forward wavefield, particle velocity, source and receivers
    save_t = geometry.nt if save else None

    if time_order == 1:
        v = VectorTimeFunction(name="v", grid=model.grid, time_order=time_order,
                               space_order=space_order, save=save_t)
        kwargs.update({'v': v})

    p = TimeFunction(name="p", grid=model.grid, time_order=time_order,
                     space_order=space_order, save=save_t, staggered=NODE)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, p, save=save, **kwargs)

    src_term, rec_term = src_rec(p, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4, kernel='sls', time_order=2, **kwargs):
    """
    Construct an adjoint modelling operator in a viscoacoustic medium.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : selects a visco-acoustic equation from the options below:
        sls (Standard Linear Solid) :
        1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        2nd order - Bai et al. (2014) viscoacoustic equation
        ren - Ren et al. (2014) viscoacoustic equation
        deng_mcmechan - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    """
    if time_order == 1:
        va = VectorTimeFunction(name="va", grid=model.grid, time_order=time_order,
                                space_order=space_order)
        kwargs.update({'v': va})

    pa = TimeFunction(name="pa", grid=model.grid, time_order=time_order,
                      space_order=space_order, staggered=NODE)

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, pa, forward=False, **kwargs)

    src_term, rec_term = src_rec(pa, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, kernel='sls', time_order=2,
                     save=True, **kwargs):
    """
    Construct a gradient operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Option to store the entire (unrolled) wavefield.
    kernel : selects a visco-acoustic equation from the options below:
        sls (Standard Linear Solid) :
        1st order - Blanch and Symes (1995) / Dutta and Schuster (2014)
        viscoacoustic equation
        2nd order - Bai et al. (2014) viscoacoustic equation
        ren - Ren et al. (2014) viscoacoustic equation
        deng_mcmechan - Deng and McMechan (2007) viscoacoustic equation
        Defaults to sls 2nd order.
    """
    # Gradient symbol and wavefield symbols
    save_t = geometry.nt if save else None

    grad = Function(name='grad', grid=model.grid)
    p = TimeFunction(name='p', grid=model.grid, time_order=time_order,
                     space_order=space_order, save=save_t, staggered=NODE)
    pa = TimeFunction(name='pa', grid=model.grid, time_order=time_order,
                      space_order=space_order, staggered=NODE)

    if time_order == 1:
        va = VectorTimeFunction(name="va", grid=model.grid, time_order=time_order,
                                space_order=space_order)
        kwargs.update({'v': va})

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn = eq_kernel(model, geometry, pa, forward=False, save=False, **kwargs)

    if time_order == 1:
        gradient_update = Eq(grad, grad - p.dt * pa)
    else:
        gradient_update = Eq(grad, grad + p.dt * pa.dt)

    # Add expression for receiver injection
    _, recterm = src_rec(pa, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + recterm + [gradient_update], subs=model.spacing_map,
                    name='Gradient', **kwargs)


def BornOperator(model, geometry, space_order=4, kernel='sls', time_order=2, **kwargs):
    """
    Construct an Linearized Born operator in an acoustic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    # Create wavefields and a dm field
    p = TimeFunction(name='p', grid=model.grid, time_order=time_order,
                     space_order=space_order, staggered=NODE)
    P = TimeFunction(name='P', grid=model.grid, time_order=time_order,
                     space_order=space_order, staggered=NODE)
    rp = TimeFunction(name="rp", grid=model.grid, time_order=time_order,
                      space_order=space_order, staggered=NODE)
    rP = TimeFunction(name="rP", grid=model.grid, time_order=time_order,
                      space_order=space_order, staggered=NODE)
    dm = Function(name='dm', grid=model.grid, space_order=0)

    if time_order == 1:
        v = VectorTimeFunction(name="v", grid=model.grid,
                               time_order=time_order, space_order=space_order)
        kwargs.update({'v': v})

    # Equations kernels
    eq_kernel = kernels[kernel]
    eqn1 = eq_kernel(model, geometry, p, r=rp, **kwargs)

    s = model.grid.stepping_dim.spacing

    if time_order == 1:
        dv = VectorTimeFunction(name="dv", grid=model.grid,
                                time_order=time_order, space_order=space_order)
        kwargs.update({'v': dv})

        q = -dm * (p.forward - p) / s
    else:
        q = -dm * (p.forward - 2 * p + p.backward) / (s**2)

    eqn2 = eq_kernel(model, geometry, P, r=rP, q=q, **kwargs)

    # Add source term expression for p
    src_term, _ = src_rec(p, model, geometry)

    # Create receiver interpolation expression from P
    _, rec_term = src_rec(P, model, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + src_term + rec_term + eqn2, subs=model.spacing_map,
                    name='Born', **kwargs)


kernels = {'sls': sls, 'ren': ren, 'deng_mcmechan': deng_mcmechan}
stencils = {('sls', 1): sls_1st_order, ('sls', 2): sls_2nd_order,
            ('deng_mcmechan', 1): deng_1st_order,
            ('deng_mcmechan', 2): deng_2nd_order,
            ('ren', 1): ren_1st_order, ('ren', 2): ren_2nd_order}
