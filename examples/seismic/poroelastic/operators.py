from devito import Eq, Operator, TimeFunction, NODE
from examples.seismic import PointSource, Receiver


def stress_fields(model, save, space_order):
    """
    Create the TimeFunction objects for the stress fields in the poroelastic formulation
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_xx = stagg_zz = NODE
        stagg_xz = (x, z)
        # Create symbols for forward wavefield, source and receivers
        txx = TimeFunction(name='txx', grid=model.grid, staggered=stagg_xx, save=save,
                           time_order=1, space_order=space_order)
        tzz = TimeFunction(name='tzz', grid=model.grid, staggered=stagg_zz, save=save,
                           time_order=1, space_order=space_order)
        txz = TimeFunction(name='txz', grid=model.grid, staggered=stagg_xz, save=save,
                           time_order=1, space_order=space_order)
        tyy = txy = tyz = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_xx = stagg_yy = stagg_zz = NODE
        stagg_xz = (x, z)
        stagg_yz = (y, z)
        stagg_xy = (x, y)
        # Create symbols for forward wavefield, source and receivers
        txx = TimeFunction(name='txx', grid=model.grid, staggered=stagg_xx, save=save,
                           time_order=1, space_order=space_order)
        tzz = TimeFunction(name='tzz', grid=model.grid, staggered=stagg_zz, save=save,
                           time_order=1, space_order=space_order)
        tyy = TimeFunction(name='tyy', grid=model.grid, staggered=stagg_yy, save=save,
                           time_order=1, space_order=space_order)
        txz = TimeFunction(name='txz', grid=model.grid, staggered=stagg_xz, save=save,
                           time_order=1, space_order=space_order)
        txy = TimeFunction(name='txy', grid=model.grid, staggered=stagg_xy, save=save,
                           time_order=1, space_order=space_order)
        tyz = TimeFunction(name='tyz', grid=model.grid, staggered=stagg_yz, save=save,
                           time_order=1, space_order=space_order)

    return txx, tyy, tzz, txy, txz, tyz


def pressure_fields(model, save, space_order):
    """
    Create the TimeFunction objects for the pressure fields in the poroelastic formulation
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_p = NODE
        # Create symbols for forward wavefield, source and receivers
        p = TimeFunction(name='p', grid=model.grid, staggered=stagg_p, save=save,
                           time_order=1, space_order=space_order)
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_p = NODE
        # Create symbols for forward wavefield, source and receivers
        p = TimeFunction(name='p', grid=model.grid, staggered=stagg_p, save=save,
                           time_order=1, space_order=space_order)
    return p


def particle_velocity_fields(model, save, space_order):
    """
    Create the particle velocity fields
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_x = x
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order, save=save)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order, save=save)
        vy = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_x = x
        stagg_y = y
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        vx = TimeFunction(name='vx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order, save=save)
        vy = TimeFunction(name='vy', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order, save=save)
        vz = TimeFunction(name='vz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order, save=save)

    return vx, vy, vz
# ------------------------------------------------------------------------------

def relative_velocity_fields(model, save, space_order):
    """
    Create the relative velocity fields
    """
    if model.grid.dim == 2:
        x, z = model.space_dimensions
        stagg_x = x
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        qx = TimeFunction(name='qx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order, save=save)
        qz = TimeFunction(name='qz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order, save=save)
        qy = None
    elif model.grid.dim == 3:
        x, y, z = model.space_dimensions
        stagg_x = x
        stagg_y = y
        stagg_z = z
        # Create symbols for forward wavefield, source and receivers
        qx = TimeFunction(name='qx', grid=model.grid, staggered=stagg_x,
                          time_order=1, space_order=space_order, save=save)
        qy = TimeFunction(name='qy', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order, save=save)
        qz = TimeFunction(name='qz', grid=model.grid, staggered=stagg_z,
                          time_order=1, space_order=space_order, save=save)

    return qx, qy, qz
# ------------------------------------------------------------------------------

def poroelastic_2d(model, space_order, save, geometry):
    """
    2D poroelastic wave equation FD kernel
    """
    rhos        = model.rhos
    rhof        = model.rhof
    phi         = model.phi
    fvs         = model.fvs
    K_dr        = model.kdr
    K_f         = model.kfl
    K_s         = model.ksg
    mu          = model.shm
    prm         = model.prm
    T           = model.T
    damp        = model.damp
    
    # Derived parameters  
  
    # Bulk Density
    rhob = phi*rhof + (1.0-phi)*rhos

    # Effective fluid density / mass coupling coefficient, kg/m**3
    rhom = T * (rhof/phi)  
    
    
    # Biot Coefficient
    alpha = 1.0 - K_dr/K_s
    
    # Biot modulus
    biotmod = 1.0/((alpha - phi)/K_s + phi/K_f)
    
    # Lame parameter for solid matrix
    l = K_dr - 2.0*mu/3.0

    # Delta T (sic)
    dt = model.grid.stepping_dim.spacing #/ 1000.0   # ns ==> s
    #dt = model.critical_dt                  # s

    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = particle_velocity_fields(model, save, space_order)
    qx, qy, qz = relative_velocity_fields(model, save, space_order)
    txx, tyy, tzz, txy, txz, tyz = stress_fields(model, save, space_order)
    p = pressure_fields(model, save, space_order) # Different order needed?


    # Account for inertia
    c0 = (rhob*rhom - rhof*rhof) / dt
    c1 = c0 + rhob*(fvs/prm)*0.5
    c2 = c0 - rhob*(fvs/prm)*0.5
    
    # Stencils
        
    # Calculate fluid pressure, then stresses
    u_p   = Eq(p.forward,   damp * ( p   + ( -1.0*alpha*biotmod * (vx.dx + vz.dy) - biotmod * (qx.dx + qz.dy) ) * dt ) )

    u_txz = Eq(txz.forward, damp * ( txz +  mu * (vx.dy + vz.dx) * dt ) )
    u_txx = Eq(txx.forward, damp * ( txx + ( (l + 2*mu)*(vx.dx) + l*vz.dy - alpha* ( -1.0*alpha*biotmod * (vx.dx + vz.dy) - biotmod * (qx.dx + qz.dy) ) ) * dt ) )
    u_tzz = Eq(txx.forward, damp * ( tzz + ( (l + 2*mu)*(vz.dy) + l*vx.dx - alpha* ( -1.0*alpha*biotmod * (vx.dx + vz.dy) - biotmod * (qx.dx + qz.dy) ) ) * dt ) )
    
    # Add sources (Use pressure / stress source)
    src_rec_expr = src_rec(vx, vy, vz, qx, qy, qz, txx, tyy, tzz, p, model, geometry)    

    # Relative velocities
    u_qx = Eq(qx.forward, damp * (c2*qx + ( -1.0*rhof*(txx.forward.dx + txz.forward.dy) - rhob*p.forward.dx )) / c1 )
    u_qz = Eq(qz.forward, damp * (c2*qz + ( -1.0*rhof*(txz.forward.dx + tzz.forward.dy) - rhob*p.forward.dy )) / c1 )

    # Matrix velocities
    u_vx = Eq(vx.forward, damp * (vx + ( rhom*(txx.forward.dx + txz.forward.dy) ) + rhof*(p.forward.dx) + rhof*(fvs/prm)*(qx + qx.forward)*0.5 ) / c0 )
    u_vz = Eq(vz.forward, damp * (vz + ( rhom*(txz.forward.dx + tzz.forward.dy) ) + rhof*(p.forward.dy) + rhof*(fvs/prm)*(qz + qz.forward)*0.5 ) / c0 )

    return [u_vx, u_vz, u_qx, u_qz, u_txx, u_tzz, u_txz, u_p] + src_rec_expr
# ------------------------------------------------------------------------------

def poroelastic_3d(model, space_order, save, geometry):
    """
    3D elastic wave equation FD kernel
    """
    vp, vs, rho_s, rho_f, phi, k, mu_f, K_dr, K_s, K_f, damp = model.vp, model.vs, model.rho_s, model.rho_f, model.phi, model.k, model.mu_f, model.K_dr, model.K_s, model.K_f, model.damp

    # Delta T (sic)
    dt = model.grid.stepping_dim.spacing    # s

    # Biot Coefficient
    alpha = 1.0 - K_dr/K_s

    # Biot Modulus
    M =  (phi/K_f + (alpha - phi)/K_s)**-1

    # Bulk Density
    rho_b = phi*rho_f + (1.0 - phi)*rho_s

    # Shear Modulus of Saturated Rock
    mu = (vs**2)*rho_b

    # Lame Parameter of Saturated Rock
    l = rho_b*(vp**2 - 2*(vs**2))

    # Create symbols for forward wavefield, source and receivers
    vx, vy, vz = particle_velocity_fields(model, save, space_order)
    qx, qy, qz = relative_velocity_fields(model, save, space_order)
    txx, tyy, tzz, txy, txz, tyz = stress_fields(model, save, space_order)

    # Stencils
    u_vx = Eq(vx.forward, damp * vx - damp * dt * 1.0/rho_b * (txx.dx + txy.dy + txz.dz))
    u_vy = Eq(vy.forward, damp * vy - damp * dt * 1.0/rho_b * (txy.dx + tyy.dy + tyz.dz))
    u_vz = Eq(vz.forward, damp * vz - damp * dt * 1.0/rho_b * (txz.dx + tyz.dy + tzz.dz))

    u_txx = Eq(txx.forward, damp * txx - damp * (l + 2 * mu) * dt * vx.forward.dx
                                       - damp * l * dt * (vy.forward.dy + vz.forward.dz))
    u_tyy = Eq(tyy.forward, damp * tyy - damp * (l + 2 * mu) * dt * vy.forward.dy
                                       - damp * l * dt * (vx.forward.dx + vz.forward.dz))
    u_tzz = Eq(tzz.forward, damp * tzz - damp * (l+2*mu)*dt * vz.forward.dz
                                       - damp * l * dt * (vx.forward.dx + vy.forward.dy))
    u_txz = Eq(txz.forward, damp * txz - damp * mu * dt * (vx.forward.dz + vz.forward.dx))
    u_txy = Eq(txy.forward, damp * txy - damp * mu * dt * (vy.forward.dx + vx.forward.dy))
    u_tyz = Eq(tyz.forward, damp * tyz - damp * mu * dt * (vy.forward.dz + vz.forward.dy))

    src_rec_expr = src_rec(vx, vy, vz, txx, tyy, tzz, model, geometry)
    return [u_vx, u_vy, u_vz, u_txx, u_tyy, u_tzz, u_txz, u_txy, u_tyz] + src_rec_expr
# ------------------------------------------------------------------------------

def src_rec(vx, vy, vz, qx, qy, qz, txx, tyy, tzz, p, model, geometry):
    """
    Source injection and receiver interpolation
    """
    dt = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec1 = Receiver(name='rec1', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)
    rec2 = Receiver(name='rec2', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)
    M = model.M
    # The source injection term
    #src_xx = src.inject(field=txx.forward, expr=src * (1.0 - model.phi) * dt, offset=model.nbpml)
    #src_zz = src.inject(field=tzz.forward, expr=src * (1.0 - model.phi) * dt, offset=model.nbpml)
    #src_xx = src.inject(field=txx.forward, expr=src * dt)
    #src_zz = src.inject(field=tzz.forward, expr=src * dt)
    src_pp = src.inject(field=p.forward,   expr=src * M)
    #src_expr = src_xx + src_zz + src_pp    
    src_expr = src_pp
    if model.grid.dim == 3:
        src_yy = src.inject(field=tyy.forward, expr=src * (1.0 - model.phi) * dt, offset=model.nbpml)
        src_expr += src_yy

    # Create interpolation expression for receivers
    rec_term1 = rec1.interpolate(expr=p, offset=model.nbpml)
    if model.grid.dim == 2:
        rec_expr = vx.dx + vz.dy
    else:
        rec_expr = vx.dx + vy.dy + vz.dz
    rec_term2 = rec2.interpolate(expr=rec_expr, offset=model.nbpml)

    return src_expr + rec_term1 + rec_term2
# ------------------------------------------------------------------------------

def ForwardOperator(model, geometry, space_order=4,
                    save=False, **kwargs):
    """
    Constructor method for the forward modelling operator in a poroelastic media

    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three buffered
                 indices (last three time steps)
    """
    
    wave = kernels[model.grid.dim]
    pde = wave(model, space_order, geometry.nt if save else None, geometry)

    # Substitute spacing terms to reduce flops
    return Operator(pde, subs=model.spacing_map,
                    name='Forward', **kwargs)


kernels = {3: poroelastic_3d, 2: poroelastic_2d}
