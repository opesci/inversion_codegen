from devito import Eq, Operator, Function, TimeFunction, Inc, solve, sign
from devito.symbolics import retrieve_functions, INT
from examples.seismic import PointSource, Receiver


def freesurface(model, eq):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation.

    Parameters
    ----------
    model : Model
        Physical model.
    eq : Eq
        Time-stepping stencil (time update) to mirror at the freesurface.
    """
    lhs, rhs = eq.evaluate.args
    # Get vertical dimension and corresponding subdimension
    zfs = model.grid.subdomains['fsdomain'].dimensions[-1]
    z = zfs.parent

    # Functions present in the stencil
    funcs = retrieve_functions(rhs)
    mapper = {}
    # Antisymmetric mirror at negative indices
    # TODO: Make a proper "mirror_indices" tool function
    for f in funcs:
        zind = f.indices[-1]
        if (zind - z).as_coeff_Mul()[0] < 0:
            s = sign(zind.subs({z: zfs, z.spacing: 1}))
            mapper.update({f: s * f.subs({zind: INT(abs(zind))})})
    return Eq(lhs, rhs.subs(mapper), subdomain=model.grid.subdomains['fsdomain'])


def laplacian(field, model, kernel):
    """
    Spatial discretization for the isotropic acoustic wave equation. For a 4th
    order in time formulation, the 4th order time derivative is replaced by a
    double laplacian:
    H = (laplacian + s**2/12 laplacian(1/m*laplacian))

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    """
    if kernel not in ['OT2', 'OT4']:
        raise ValueError("Unrecognized kernel")
    s = model.grid.time_dim.spacing
    biharmonic = field.biharmonic(1/model.m) if kernel == 'OT4' else 0
    return field.laplace + s**2/12 * biharmonic


def iso_stencil(field, model, kernel, **kwargs):
    """
    Stencil for the acoustic isotropic wave-equation:
    u.dt2 - H + damp*u.dt = 0.

    Parameters
    ----------
    field : TimeFunction
        The computed solution.
    model : Model
        Physical model.
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    q : TimeFunction, Function or float
        Full-space/time source of the wave-equation.
    forward : bool, optional
        Whether to propagate forward (True) or backward (False) in time.
    """
    # Forward or backward
    forward = kwargs.get('forward', True)
    # Define time step to be updated
    unext = field.forward if forward else field.backward
    udt = field.dt if forward else field.dt.T
    # Get the spacial FD
    lap = laplacian(field, model, kernel)
    # Get source
    q = kwargs.get('q', 0)
    # Define PDE and update rule
    eq_time = solve(model.m * field.dt2 - lap - q + model.damp * udt, unext)

    # Time-stepping stencil.
    eqns = [Eq(unext, eq_time, subdomain=model.grid.subdomains['physdomain'])]

    # Add free surface
    if model.fs:
        eqns.append(freesurface(model, Eq(unext, eq_time)))
    return eqns


def ForwardOperator(model, geometry, space_order=4,
                    save=False, kernel='OT2', **kwargs):
    """
    Construct a forward modelling operator in an acoustic medium.

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
    kernel : str, optional
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid,
                     save=geometry.nt if save else None,
                     time_order=2, space_order=space_order)
    src = PointSource(name='src', grid=geometry.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=geometry.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(u, model, kernel)

    # Construct expression to inject source values
    src_term = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create interpolation expression for receivers
    rec_term = rec.interpolate(expr=u)
    # Substitute spacing terms to reduce flops
    return Operator(eqn + src_term + rec_term, subs=model.spacing_map,
                    name='Forward', **kwargs)


def AdjointOperator(model, geometry, space_order=4,
                    kernel='OT2', **kwargs):
    """
    Construct an adjoint modelling operator in an acoustic media.

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
        Type of discretization, 'OT2' or 'OT4'.
    """
    m = model.m

    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    srca = PointSource(name='srca', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    # Construct expression to inject receiver values
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Create interpolation expression for the adjoint-source
    source_a = srca.interpolate(expr=v)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + source_a, subs=model.spacing_map,
                    name='Adjoint', **kwargs)


def GradientOperator(model, geometry, space_order=4, save=True,
                     kernel='OT2', **kwargs):
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
    kernel : str, optional
        Type of discretization, centered or shifted.
    """
    m = model.m

    # Gradient symbol and wavefield symbols
    grad = Function(name='grad', grid=model.grid)
    u = TimeFunction(name='u', grid=model.grid, save=geometry.nt if save
                     else None, time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    s = model.grid.stepping_dim.spacing
    eqn = iso_stencil(v, model, kernel, forward=False)

    if kernel == 'OT2':
        gradient_update = Inc(grad, - u.dt2 * v)
    elif kernel == 'OT4':
        gradient_update = Inc(grad, - (u.dt2 + s**2 / 12.0 * u.biharmonic(m**(-2))) * v)
    # Add expression for receiver injection
    receivers = rec.inject(field=v.backward, expr=rec * s**2 / m)

    # Substitute spacing terms to reduce flops
    return Operator(eqn + receivers + [gradient_update], subs=model.spacing_map,
                    name='Gradient', **kwargs)


def BornOperator(model, geometry, space_order=4,
                 kernel='OT2', **kwargs):
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
    m = model.m

    # Create source and receiver symbols
    src = Receiver(name='src', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nsrc)

    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # Create wavefields and a dm field
    u = TimeFunction(name="u", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    U = TimeFunction(name="U", grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    dm = Function(name="dm", grid=model.grid, space_order=0)

    s = model.grid.stepping_dim.spacing
    eqn1 = iso_stencil(u, model, kernel)
    eqn2 = iso_stencil(U, model, kernel, q=-dm*u.dt2)

    # Add source term expression for u
    source = src.inject(field=u.forward, expr=src * s**2 / m)

    # Create receiver interpolation expression from U
    receivers = rec.interpolate(expr=U)

    # Substitute spacing terms to reduce flops
    return Operator(eqn1 + source + eqn2 + receivers, subs=model.spacing_map,
                    name='Born', **kwargs)
