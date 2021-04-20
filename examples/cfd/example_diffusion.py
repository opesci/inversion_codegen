"""
NOTE: This is a legacy example. For more up-to-date examples
please see the tutorials under `examples/cfd/`.

This example encodes multiple ways to solve the 2D diffusion equations
using an explicit finite difference scheme with fixed boundary values
and a given initial value for the density. The example also includes
demo implementations using more traditional methods, including pure
Python, vectorized NumPy and a vectorized symbolic implementation
using SymPy's `lambdify()` functionality. For a visual demonstration
and comparison between pure Python and NumPy, for example, run:

python example_diffusion.py run -m python numpy --show
"""
import time
from argparse import ArgumentParser

import numpy as np
import sympy

from devito import Grid, Eq, Operator, TimeFunction, solve
from devito.logger import log

try:
    from devitobench import Benchmark, Executor, Plotter
except:
    Benchmark = None
    Executor = None
    Plotter = None


def ring_initial(spacing=0.01):
    """Initialise grid with initial condition ("ring")"""
    nx, ny = int(1 / spacing), int(1 / spacing)
    xx, yy = np.meshgrid(np.linspace(0., 1., nx, dtype=np.float32),
                         np.linspace(0., 1., ny, dtype=np.float32))
    ui = np.zeros((nx, ny), dtype=np.float32)
    r = (xx - .5)**2. + (yy - .5)**2.
    ui[np.logical_and(.05 <= r, r <= .1)] = 1.
    return ui


def execute_python(ui, spacing=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using pure Python list indexing."""
    nx, ny = ui.shape
    dx2, dy2 = spacing**2, spacing**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.concatenate((ui, np.zeros_like(ui))).reshape((2, nx, ny))

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        t0 = ti % 2
        t1 = (ti + 1) % 2
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                uxx = (u[t0, i+1, j] - 2*u[t0, i, j] + u[t0, i-1, j]) / dx2
                uyy = (u[t0, i, j+1] - 2*u[t0, i, j] + u[t0, i, j-1]) / dy2
                u[t1, i, j] = u[t0, i, j] + dt * a * (uxx + uyy)
    runtime = time.time() - tstart
    log("Python: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"
        % (spacing, spacing, timesteps, runtime))
    return u[ti % 2, :, :], runtime


def execute_numpy(ui, spacing=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using vectorised numpy array accesses."""
    nx, ny = ui.shape
    dx2, dy2 = spacing**2, spacing**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.concatenate((ui, np.zeros_like(ui))).reshape((2, nx, ny))

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        t0 = ti % 2
        t1 = (ti + 1) % 2

        uxx = (u[t0, 2:, 1:-1] - 2*u[t0, 1:-1, 1:-1] + u[t0, :-2, 1:-1]) / dx2
        uyy = (u[t0, 1:-1, 2:] - 2*u[t0, 1:-1, 1:-1] + u[t0, 1:-1, :-2]) / dy2
        u[t1, 1:-1, 1:-1] = u[t0, 1:-1, 1:-1] + a * dt * (uxx + uyy)
    runtime = time.time() - tstart
    log("Numpy: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"
        % (spacing, spacing, timesteps, runtime))
    return u[ti % 2, :, :], runtime


def execute_lambdify(ui, spacing=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using vectorised numpy array accesses."""
    nx, ny = ui.shape
    dx2, dy2 = spacing**2, spacing**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    u = np.concatenate((ui, np.zeros_like(ui))).reshape((2, nx, ny))

    def diffusion_stencil():
        """Create stencil and substitutions for the diffusion equation"""
        p = sympy.Function('p')
        x, y, t, h, s = sympy.symbols('x y t h s')
        dx2 = p(x, y, t).diff(x, x).as_finite_difference([x - h, x, x + h])
        dy2 = p(x, y, t).diff(y, y).as_finite_difference([y - h, y, y + h])
        dt = p(x, y, t).diff(t).as_finite_difference([t, t + s])
        eqn = Eq(dt, a * (dx2 + dy2))
        stencil = solve(eqn, p(x, y, t + s))
        return stencil, (p(x, y, t), p(x + h, y, t), p(x - h, y, t),
                         p(x, y + h, t), p(x, y - h, t), s, h)
    stencil, subs = diffusion_stencil()
    kernel = sympy.lambdify(subs, stencil, 'numpy')

    # Execute timestepping loop with alternating buffers
    tstart = time.time()
    for ti in range(timesteps):
        t0 = ti % 2
        t1 = (ti + 1) % 2
        u[t1, 1:-1, 1:-1] = kernel(u[t0, 1:-1, 1:-1], u[t0, 2:, 1:-1],
                                   u[t0, :-2, 1:-1], u[t0, 1:-1, 2:],
                                   u[t0, 1:-1, :-2], dt, spacing)
    runtime = time.time() - tstart
    log("Lambdify: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"
        % (spacing, spacing, timesteps, runtime))
    return u[ti % 2, :, :], runtime


def execute_devito(ui, spacing=0.01, a=0.5, timesteps=500):
    """Execute diffusion stencil using the devito Operator API."""
    nx, ny = ui.shape
    dx2, dy2 = spacing**2, spacing**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))
    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    grid = Grid(shape=(nx, ny))
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2)
    u.data[0, :] = ui[:]

    # Derive the stencil according to devito conventions
    eqn = Eq(u.dt, a * (u.dx2 + u.dy2))
    stencil = solve(eqn, u.forward)
    op = Operator(Eq(u.forward, stencil))

    # Execute the generated Devito stencil operator
    tstart = time.time()
    op.apply(u=u, t=timesteps, dt=dt)
    runtime = time.time() - tstart
    log("Devito: Diffusion with dx=%0.4f, dy=%0.4f, executed %d timesteps in %f seconds"
        % (spacing, spacing, timesteps, runtime))
    return u.data[1, :], runtime


def animate(field):
    """Animate solution field"""
    import matplotlib.pyplot as plt
    xvals = np.linspace(0, 1., field.shape[0])
    yvals = np.linspace(0, 1., field.shape[1])
    levels = np.linspace(-.1, 1.1, 21)
    cs = plt.contourf(xvals, yvals, field, levels=levels)
    plt.colorbar(cs)
    plt.show()


def test_diffusion2d(spacing=0.01, timesteps=1000):
    ui = ring_initial(spacing=spacing)
    u, _ = execute_devito(ui, spacing=spacing, timesteps=timesteps)
    assert(u.max() < 2.4)
    assert(np.linalg.norm(u, ord=2) < 13)


if __name__ == "__main__":
    description = """
Example script demonstrating several approaches to solving the
2D diffusion equation. The various modes are pure Python,
vectorized numpy, a lambdified SymPy equation and the Devito API.

Please note that the default settings for spacing and timesteps are
chosen to highlight that Devito and "vectorised numpy" are faster than
pure Python or the sympy.lambdify kernels approach.  For a fair
performance comparison between Devito and "vectorised numpy" we
recommend using --spacing 0.001 -t 1000.
"""
    parser = ArgumentParser(description=description)
    parser.add_argument(dest='execmode', nargs='?', default='run',
                        choices=['run', 'bench', 'plot'],
                        help="Script mode; either 'run', 'bench' or 'plot' ")
    parser.add_argument('-m', '--mode', nargs='+', default=['devito'],
                        choices=['python', 'numpy', 'lambdify', 'devito'],
                        help="Example modes: python, numpy, lambdify, devito")
    parser.add_argument('-s', '--spacing', type=float, nargs='+', default=[0.01],
                        help='Grid spacing in x and y)')
    parser.add_argument('-t', '--timesteps', type=int, default=20,
                        help='Number of timesteps to run')
    parser.add_argument('--show', action='store_true',
                        help="Show animation of the solution field")
    parser.add_argument('-i', '--resultsdir', default='results',
                        help='Directory containing results')
    parser.add_argument('-o', '--plotdir', default='plots',
                        help='Directory to store generated plots')

    args = parser.parse_args()
    if len(args.spacing) > 2:
        raise ValueError("Too many arguments encountered for --spacing")

    # Map "mode" arg to execution function
    exec_func = {'python': execute_python, 'numpy': execute_numpy,
                 'lambdify': execute_lambdify, 'devito': execute_devito}
    # Create parameters dict and remove script-specific args
    parameters = vars(args).copy()
    del parameters['show']
    del parameters['execmode']
    del parameters['resultsdir']
    del parameters['plotdir']

    if args.execmode == 'run':
        if len(args.spacing) > 1:
            raise ValueError("Run mode only supports a single --spacing value")

        # Run and animate example runs
        for mode in args.mode:
            ui = ring_initial(spacing=args.spacing[0])
            if args.show:
                animate(ui)
            u, _ = exec_func[mode](ui, spacing=args.spacing[0],
                                   timesteps=args.timesteps)
            if args.show:
                animate(u)

    elif args.execmode == 'bench':
        if Benchmark is None:
            raise ImportError("Could not find devitobench utility package. Please \n"
                              "install from https://github.com/devitocodes/devitobench")

        class DiffusionExecutor(Executor):
            """Executor class that defines how to run the benchmark"""
            def setup(self, **kwargs):
                self.ui = ring_initial(spacing=kwargs['spacing'])

            def run(self, *args, **kwargs):
                u, time = exec_func[kwargs['mode']](self.ui, spacing=kwargs['spacing'],
                                                    timesteps=kwargs['timesteps'])
                self.register(time)

        # Run benchmark across parameters and save the result
        bench = Benchmark(name='Diffusion', resultsdir=args.resultsdir,
                          parameters=parameters)
        bench.execute(DiffusionExecutor(), warmups=0, repeats=1)
        bench.save()

    elif args.execmode == 'plot':
        # Load previously generated benchmark data
        bench = Benchmark(name='Diffusion', resultsdir=args.resultsdir,
                          parameters=parameters)
        bench.load()

        # Generate the plot from loaded benchmark data
        plotter = Plotter()
        plotter.plot_comparison('DiffusionModes.pdf', args.mode, bench.lookup())
