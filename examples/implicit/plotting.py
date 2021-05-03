from matplotlib import cm
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_iterative']


def plot_iterative(solution, xmin=0., xmax=1., ymin=0., ymax=1., view=None,
                   dpi=200, linewidth=0, dynamic=False, xlabel="x", ylabel="y",
                   zlabel="z", title="", titlepad=270):
    sol = solution[0, :, :]
    niter = solution.shape[0]
    if dynamic:
        zmin, zmax = np.min(sol), np.max(sol)
    else:
        zmin, zmax = np.min(solution), np.max(solution)
    X, Y = np.meshgrid(np.linspace(xmin, xmax, sol.shape[0]),
                       np.linspace(ymin, ymax, sol.shape[1]), indexing='ij')
    figure = plt.figure()
    axes = figure.gca(projection='3d')
    axes.set_zlim(zmin, zmax)
    if view is not None:
        axes.view_init(*view)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_zlabel(zlabel)
    plt.title(title)
    axes.plot_surface(X, Y, sol[:], cmap=cm.viridis, color='red',
                      rstride=2, cstride=2, linewidth=linewidth, antialiased=False)
    slider_axes = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(slider_axes, 'Iteration', 0, niter, valinit=0)

    def update(value):
        iteration = int(value)
        if iteration < 0:
            iteration = 0
        elif iteration > niter:
            iteration = niter - 1
        sol = solution[iteration, :, :]
        if dynamic:
            zmin, zmax = np.min(sol), np.max(sol)
        else:
            zmin, zmax = np.min(solution), np.max(solution)
        slider.valtext.set_text('{}'.format(iteration))
        axes.clear()
        axes.set_zlim(zmin, zmax)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_zlabel(zlabel)
        plt.title(title, pad=titlepad)
        axes.plot_surface(X, Y, sol, cmap=cm.viridis,
                          color='red', rstride=2, cstride=2,
                          linewidth=linewidth, antialiased=False)
        figure.canvas.draw()
    slider.on_changed(update)

    plt.show()
