import numpy as np
import pytest

from devito.logger import info
from devito import norm
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver
from examples.seismic import demo_model, setup_geometry, seismic_args


def viscoacoustic_setup(shape=(50, 50), spacing=(15.0, 15.0), tn=500., space_order=4,
                        nbl=40, preset='layers-viscoacoustic', kernel='sls',
                        time_order=2, **kwargs):
    model = demo_model(preset, space_order=space_order, shape=shape, nbl=nbl,
                       dtype=kwargs.pop('dtype', np.float32), spacing=spacing)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = ViscoacousticWaveSolver(model, geometry, space_order=space_order,
                                     kernel=kernel, time_order=time_order, **kwargs)
    return solver


def run(shape=(50, 50), spacing=(20.0, 20.0), tn=1000.0,
        space_order=4, nbl=40, autotune=False, preset='layers-viscoacoustic',
        kernel='sls', time_order=2, **kwargs):

    solver = viscoacoustic_setup(shape=shape, spacing=spacing, nbl=nbl, tn=tn,
                                 space_order=space_order, preset=preset,
                                 kernel=kernel, time_order=time_order, **kwargs)
    info("Applying Forward")

    rec, p, v, summary = solver.forward(autotune=autotune)

    return (summary.gflopss, summary.oi, summary.timings, [rec, p, v])


@pytest.mark.parametrize('kernel, time_order, normrec, atol', [
    ('sls', 2, 684.385, 1e-2),
    ('sls', 1, 42.243, 1e-2),
    ('ren', 2, 677.673, 1e-2),
    ('ren', 1, 40.489, 1e-2),
    ('deng_mcmechan', 2, 673.041, 1e-2),
    ('deng_mcmechan', 1, 41.598, 1e-2),
])
def test_viscoacoustic(kernel, time_order, normrec, atol):
    _, _, _, [rec, _, _] = run(kernel=kernel, time_order=time_order)
    assert np.isclose(norm(rec), normrec, atol=atol, rtol=0)


@pytest.mark.parametrize('ndim', [2, 3])
@pytest.mark.parametrize('kernel', ['sls', 'ren', 'deng_mcmechan'])
@pytest.mark.parametrize('time_order', [1, 2])
def test_viscoacoustic_stability(ndim, kernel, time_order):
    shape = tuple([11]*ndim)
    spacing = tuple([20]*ndim)
    _, _, _, [rec, _, _] = run(shape=shape, spacing=spacing, tn=20000.0, nbl=0,
                               kernel=kernel, time_order=time_order)
    assert np.isfinite(norm(rec))


if __name__ == "__main__":
    description = ("Example script for a set of viscoacoustic operators.")
    parser = seismic_args(description)
    parser.add_argument("-k", dest="kernel", default='sls',
                        choices=['sls', 'ren', 'deng_mcmechan'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-to", "--time_order", default=2,
                        type=int, help="Time order of the equation")
    args = parser.parse_args()
    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [10.0])
    tn = args.tn if args.tn > 0 else (750. if ndim < 3 else 1250.)
    preset = 'constant-viscoacoustic' if args.constant else 'layers-viscoacoustic'

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn, opt=args.opt,
        space_order=args.space_order, autotune=args.autotune, preset=preset,
        kernel=args.kernel, time_order=args.time_order)
