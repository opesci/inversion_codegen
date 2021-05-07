from devito import Eq, Operator, Function, TimeFunction


def iso_stencil(field, model, **kwargs):
    """
    Stencil for the scalar isotropic visco- acoustic variable density
    self adjoint wave equation:

        b/v^2 [ P.dt2 + w/Q P.dt ] = (b P.dx).dx + (b P.dy).dy + (b P.dz).dz + s

    Note derivative shifts are omitted for simplicity above.
    See implementation notebook sa_01_iso_implementation1.ipynb for more details.

    Parameters
    ----------
    field : TimeFunction, required
        The pressure wavefield computed solution.
    model : Dictionary <string>:<Function>, contains:
        'b': Buoyancy = reciprocal density (units: m^3/kg)
        'v': Velocity (units: m/msec or km/sec)
        'wOverQ': The w/Q field for dissipation only attenuation.
    forward : bool, optional
        The propagation direction. Defaults to True.
    q : TimeFunction, Function or float, optional
        Full-space/time source of the wave-equation.

    Returns
    ----------
    The time update stencil.
    """
    # Get the Functions for buoyancy, velocity, and wOverQ
    vp, b, wOverQ = model.vp, model.b, model.damp

    # Define time step of pressure wavefield to be updated
    forward = kwargs.get('forward', True)

    if forward:
        field_next = field.forward
        field_prev = field.backward
    else:
        field_next = field.backward
        field_prev = field.forward

    # Get the source
    q = kwargs.get('q', 0)

    # Define the time update equation for 2d/3d
    if len(field.data.shape) == 3:
        t, x, y = field.dimensions
        eq_time_update = (t.spacing**2 * vp**2 / b) * \
            ((b * field.dx(x0=x+x.spacing/2)).dx(x0=x-x.spacing/2) +
             (b * field.dy(x0=y+y.spacing/2)).dy(x0=y-y.spacing/2) + q) + \
            (2 - t.spacing * wOverQ) * field + \
            (t.spacing * wOverQ - 1) * field_prev

    else:
        t, x, y, z = field.dimensions
        eq_time_update = (t.spacing**2 * vp**2 / b) * \
            ((b * field.dx(x0=x+x.spacing/2)).dx(x0=x-x.spacing/2) +
             (b * field.dy(x0=y+y.spacing/2)).dy(x0=y-y.spacing/2) +
             (b * field.dz(x0=z+z.spacing/2)).dz(x0=z-z.spacing/2) + q) + \
            (2 - t.spacing * wOverQ) * field + \
            (t.spacing * wOverQ - 1) * field_prev

    return [Eq(field_next, eq_time_update)]


def IsoFwdOperator(model, geometry, space_order=8, save=False, **kwargs):
    """
    Construct a forward modeling Operator in a variable density visco- acoustic media.
    See implementation notebook sa_01_iso_implementation1.ipynb for more details.

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
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.

    Returns
    ----------
    The Operator implementing forward modeling.
    """
    src = geometry.src
    rec = geometry.rec
    vp, b = model.vp, model.b
    # Create symbols for wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)

    # Time update equation
    eqn = iso_stencil(u, model, forward=True)

    # Construct expression to inject source values, injecting at p(t+dt)
    t = u.grid.time_dim
    src_term = src.inject(field=u.forward, expr=src * t.spacing**2 * vp**2 / b)

    # Create interpolation expression for receivers, extracting at p(t)
    rec_term = rec.interpolate(expr=u)

    # Substitute spacing terms to reduce flops
    spacing_map = model.spacing_map

    kwargs['opt'] = ('advanced', {'cire-mingain': 1000})

    return Operator(eqn + src_term + rec_term, subs=spacing_map,
                    name='IsoFwdOperator', **kwargs)


def IsoAdjOperator(model, geometry, space_order=8, save=False, **kwargs):
    """
    Construct an adjoint modeling Operator in a variable density visco- acoustic media.
    Note the FD evolution will be time reversed.
    See implementation notebook sa_01_iso_implementation1.ipynb for more details.

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
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.

    Returns
    ----------
    The Operator implementing adjoint modeling.
    """
    rec = geometry.rec
    src = geometry.src
    vp, b = model.vp, model.b
    # Create symbols for wavefield, source and receivers
    v = TimeFunction(name='v', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)

    # Time update equation
    eqn = iso_stencil(v, model, forward=False)

    # Construct expression to inject receiver values, injecting at p(t-dt)
    t = model.grid.time_dim
    rec_term = rec.inject(field=v.backward, expr=rec * t.spacing**2 * vp**2 / b)

    # Create interpolation expression for the adjoint-source, extracting at p(t)
    src_term = src.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    spacing_map = model.spacing_map

    return Operator(eqn + rec_term + src_term, subs=spacing_map,
                    name='IsoAdjOperator', **kwargs)


def IsoJacobianFwdOperator(model, geometry, space_order=8,
                           save=False, **kwargs):
    """
    Construct a linearized JacobianForward modeling Operator in a variable density
    visco- acoustic media.

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
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.

    Returns
    ----------
    The Operator implementing Jacobian forward modeling.
    """
    src = geometry.src
    rec = geometry.rec
    vp, b, wOverQ = model.vp, model.b, model.damp
    # Create p0, dp wavefields and dm velocity perturbation field
    u0 = TimeFunction(name="u0", grid=model.grid,
                      save=geometry.nt if save else None,
                      time_order=2, space_order=space_order)

    du = TimeFunction(name="du", grid=model.grid,
                      time_order=2, space_order=space_order)

    dm = Function(name="dm", grid=model.grid, space_order=space_order)

    # Time update equations
    # JKW: this is pretty cool, simultaneously solving for p0 and dp!
    # The 1st equation is derived in sa_01_iso_implementation1.ipynb
    # The 2nd equation is derived in sa_02_iso_implementation2.ipynb
    t = u0.time_dim
    eqn1 = iso_stencil(u0, model, forward=True)

    # Linearized source and stencil
    lin_src = 2 * b * dm * vp**-3 * (wOverQ * u0.dt(x0=t-t.spacing/2) + u0.dt2)
    eqn2 = iso_stencil(du, model, forward=True, q=lin_src)

    # Construct expression to inject source values, injecting at p0(t+dt)
    src_term = src.inject(field=u0.forward, expr=src * t.spacing**2 * vp**2 / b)

    # Create interpolation expression for receivers, extracting at dp(t)
    rec_term = rec.interpolate(expr=du)

    # Substitute spacing terms to reduce flops
    spacing_map = model.spacing_map

    return Operator(eqn1 + src_term + eqn2 + rec_term, subs=spacing_map,
                    name='IsoJacobianFwdOperator', **kwargs)


def IsoJacobianAdjOperator(model, geometry, space_order=8,
                           save=True, **kwargs):
    """
    Construct a linearized JacobianAdjoint modeling Operator in a variable density
    visco- acoustic media.

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
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.

    Returns
    ----------
    The Operator implementing Jacobian adjoint modeling.
    """
    rec = geometry.rec
    vp, b, wOverQ = model.vp, model.b, model.damp
    # Create p0, dp wavefields and dm velocity perturbation field
    u0 = TimeFunction(name="u0", grid=model.grid,
                      save=geometry.nt if save else None,
                      time_order=2, space_order=space_order)

    du = TimeFunction(name="du", grid=model.grid,
                      time_order=2, space_order=space_order)

    dm = Function(name="dm", grid=model.grid, space_order=space_order)

    # Time update equation
    t = u0.time_dim
    eqn = iso_stencil(du, model, forward=False)
    dm_update = Eq(dm, dm +
                   du * (2 * b * vp**-3 * (wOverQ * u0.dt(x0=t-t.spacing/2) + u0.dt2)))

    # Construct expression to inject receiver values, injecting at p(t-dt)
    rec_term = rec.inject(field=du.backward, expr=rec * t.spacing**2 * vp**2 / b)

    # Substitute spacing terms to reduce flops
    spacing_map = model.spacing_map

    return Operator([dm_update] + eqn + rec_term, subs=spacing_map,
                    name='IsoJacobianAdjOperator', **kwargs)
