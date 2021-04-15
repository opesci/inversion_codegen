import numpy as np
import pytest

from devito import info, norm

from examples.seismic import demo_model, setup_geometry, seismic_args
from examples.seismic.tti import AnisotropicWaveSolver


def tti_setup(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              kernel='centered', space_order=4, nbl=10, preset='layers-tti',
              **kwargs):

    # Two layer model for true velocity
    model = demo_model(preset, shape=shape, spacing=spacing,
                       space_order=space_order, nbl=nbl, **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    return AnisotropicWaveSolver(model, geometry, space_order=space_order,
                                 kernel=kernel, **kwargs)


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        autotune=False, time_order=2, space_order=4, nbl=10,
        kernel='centered', full_run=False, **kwargs):

    solver = tti_setup(shape=shape, spacing=spacing, tn=tn, space_order=space_order,
                       nbl=nbl, kernel=kernel, **kwargs)
    info("Applying Forward")

    rec, u, v, summary = solver.forward(autotune=autotune)

    if not full_run:
        return summary.gflopss, summary.oi, summary.timings, [rec, u, v]

    info("Applying Adjoint")
    solver.adjoint(rec, autotune=autotune)
    return summary.gflopss, summary.oi, summary.timings, [rec, u, v]


@pytest.mark.parametrize('kernel', ['centered', 'staggered'])
@pytest.mark.parametrize('ndim', [2, 3])
def test_tti_stability(kernel, ndim):
    shape = tuple([11]*ndim)
    spacing = tuple([20]*ndim)
    _, _, _, [rec, _, _] = run(shape=shape, spacing=spacing, kernel=kernel,
                               tn=16000.0, nbl=0)
    assert np.isfinite(norm(rec))


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = seismic_args(description)
    parser.add_argument('--noazimuth', dest='azi', default=False, action='store_true',
                        help="Whether or not to use an azimuth angle")
    parser.add_argument("-k", dest="kernel", default='centered',
                        choices=['centered', 'staggered'],
                        help="Choice of finite-difference kernel")
    args = parser.parse_args()

    # Switch to TTI kernel if input is acoustic kernel
    preset = 'layers-tti-noazimuth' if args.azi else 'layers-tti'

    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [20.0])
    tn = args.tn if args.tn > 0 else (750. if ndim < 3 else 1250.)

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn,
        space_order=args.space_order, autotune=args.autotune, dtype=args.dtype,
        opt=args.opt, kernel=args.kernel, preset=preset, full_run=args.full)
