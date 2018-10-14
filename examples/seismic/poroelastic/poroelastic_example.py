import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from examples.seismic.poroelastic import PoroelasticWaveSolver
from examples.seismic import RickerSource, Receiver, TimeAxis, demo_model, plot_image
# ==============================================================================

def poroelastic_setup(shape=(50, 50), spacing=(15.0, 15.0), tn=500., space_order=4, nbpml=10,
                  constant=True, **kwargs):

    nrec = 2*shape[0]
    preset = 'constant-poroelastic' if constant else 'layers-poroelastic'
    model = demo_model(preset, space_order=space_order, shape=shape, nbpml=nbpml,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Derive timestepping from model spacing
    dt = model.critical_dt
    t0 = 0.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Define source geometry (center of domain, just below surface)
    # Ricker source peak frequency is in kHz
    src = RickerSource(name='src', grid=model.grid, f0=0.05, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    if len(shape) > 1:
        src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]
    
    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    
    if len(shape) > 1:
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]
    
    # Create solver object to provide relevant operators
    solver = PoroelasticWaveSolver(model, source=src, receiver=rec, space_order=space_order, **kwargs)
    
    return solver
# ------------------------------------------------------------------------------

def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbpml=40, autotune=False, constant=True, save=False, **kwargs):

    solver = poroelastic_setup(shape=shape, spacing=spacing, nbpml=nbpml, tn=tn,
                           space_order=space_order, constant=constant, **kwargs)
    info("Applying Forward")
    # Define receiver geometry (spread across x, just below surface)
    rec1, rec2, vx, vz, wx, wz, txx, tzz, txz, p, summary = solver.forward(autotune=autotune, save=save)
    from IPython import embed;embed()
    return rec1, rec2, vx, vz, wx, wz, txx, tzz, txz, p, summary
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    description = ("Example script for a set of poroelastic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('--2d', dest='dim2', default=True, action='store_true',
                        help="Preset to determine the physical problem setup")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLEE) mode")
    parser.add_argument("--constant", default=True, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    parser.add_argument("--save", default=False, action='store_true',
                        help="Save all wavefield output")
    args = parser.parse_args()

    # 2D preset parameters
    if args.dim2:
        shape = (1500, 1500)
        spacing = (3.0, 3.0)
        tn = 250.0
    # 3D preset parameters
    else:
        shape = (150, 150, 150)
        spacing = (10.0, 10.0, 10.0)
        tn = 1250.0

    run(shape=shape, spacing=spacing, nbpml=args.nbpml, tn=tn, dle=args.dle,
        space_order=args.space_order, autotune=args.autotune, constant=args.constant,
        dse=args.dse, save=args.save)
