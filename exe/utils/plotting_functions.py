"""
Define objects functions to plot vertex model object.
"""

from cells import init
from cells.bind import VertexModel, angle2, getPercentageKeptNeighbours,\
    getLinesHalfEdge, getLinesJunction, getPolygonsCell, hexagonEdgeLength,\
    getAllWaveVectors2D, getAllFT2D

import numpy as np
from operator import itemgetter

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm,\
    LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection, LineCollection

# PLOT VERTEX MODEL OBJECT

class WindowClosedException(Exception): pass

def _update_canvas(fig):
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.001)
    if not(plt.fignum_exists(fig.number)):  # throw error when window is closed
        raise WindowClosedException

def plot(vm, fig=None, ax=None, update=True, only_set=False,
    vertex_indices=False, cbar_zero='hexagon'):
    """
    Plot vertex model.

    Parameters
    ----------
    vm : cells.bind.VertexModel
        State of the system to plot.
    fig : matplotlib.figure.Figure or None
        Figure on which to plot. (default: None)
        NOTE: if fig == None then a new figure and axes subplot is created.
    ax : matplotlib.axes._subplots.AxesSubplot or None
        Axes subplot on which to plot. (default: None)
        NOTE: if ax == None then a new figure and axes subplot is created.
    update : bool
        Update figure canvas. (default: True)
    only_set : bool
        Only set the figure and do not plot anything. (default: False)
    vertex_indices : bool
        Write indices alongside vertices. (default: False)
    cbar_zero : defines which value to center colorbar around.  (default: 'hexagon')
        NOTE: takes 'absolute', 'hexagon', or 'average'. Only implemented for "surface"


    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure.
    ax : matplotlib.axes._subplots.AxesSubplot
        Axes subplot.
    """



    # OBTAIN MODEL PARAMETERS

    # forces

    if "area" in vm.vertexForces:
        A0 = vm.vertexForces["area"].parameters["A0"]
    if "surface" in vm.vertexForces:
        surface_force = vm.vertexForces["surface"]
        volumes = surface_force.volume
        heights = surface_force.height
        if surface_force.parameters["tauV"] == 0 or True:
            h6 = dict(map(
                lambda i: (i, volumes[i]/((np.sqrt(3)*(volumes[i]**2)/2)**(1/3))),
                volumes))
            h  = list(map(
                lambda i: heights[i], 
                heights))
            del volumes
    if "perimeter" in vm.vertexForces:
        if "area" in vm.vertexForces:
            p0 = vm.vertexForces["perimeter"].parameters["P0"]/np.sqrt(A0)
        else:
            P0 = vm.vertexForces["perimeter"].parameters["P0"]
    if "volume" in vm.vertexForces or "linear_volume" in vm.vertexForces:
        if "volume" in vm.vertexForces:
            volume_force = vm.vertexForces["volume"]
        else:
            volume_force = vm.vertexForces["linear_volume"]
        h0 = volume_force.parameters["H0"]/volume_force.parameters["A0"]
    if "boundary_tension" in vm.vertexForces:
        gamma = vm.vertexForces["boundary_tension"].parameters["gamma"]
    if "abp" in vm.vertexForces:
        v0 = vm.vertexForces["abp"].parameters["v0"]
        taup = vm.vertexForces["abp"].parameters["taup"]
    if "out" in vm.halfEdgeForces:
        t0 = vm.halfEdgeForces["out"].parameters["t0"]
        taup = vm.halfEdgeForces["out"].parameters["taup"]
    for model in ("model0", "model1"):
        if model in vm.halfEdgeForces:
            if "area" in vm.vertexForces:
                p0 = vm.halfEdgeForces[model].parameters["P0"]/np.sqrt(A0)
            else:
                P0 = vm.halfEdgeForces[model].parameters["P0"]
    for model in ("model0", "model1", "model2", "model3", "model4"):
        if model in vm.halfEdgeForces:
            sT0 = vm.halfEdgeForces[model].parameters["sigma"]
            taup = vm.halfEdgeForces[model].parameters["taup"]
    for model in ("model2", "model4"):
        if model in vm.halfEdgeForces:
            taur = vm.halfEdgeForces[model].parameters["taur"]

    # integrators

    if "eta" in vm.integrator.parameters:
        eta = vm.integrator.parameters["eta"]



    # INITIALIZE FIGURE

    def _set_lim(ax_, vm_):
        if True:
            ax_.set_xlim([0, vm_.systemSize[0]])
            ax_.set_ylim([0, vm_.systemSize[1]])
        else:
            # zoom
            n = 4
            ax_.set_xlim([vm_.systemSize[0]/n, (n - 1)*vm_.systemSize[0]/n])
            ax_.set_ylim([vm_.systemSize[0]/n, (n - 1)*vm_.systemSize[0]/n])
        ax_.set_aspect("equal")

    if fig is None or ax is None:

        plt.ioff()
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)                                         # set figure size
        try: fig.canvas.window().setFixedSize(fig.canvas.window().size())   # set window size
        except AttributeError: pass

        # all force-related colourbars
        ax_size, fig_width, fig_height = _measure_fig(ax)

        if "volume" in vm.vertexForces:
            cbar_volume = fig.colorbar(
                mappable=scalarMap_yketa, ax=ax,
                shrink=0.75, pad=0.01)
            # resize
            ax_size, fig_width, fig_height = (
                _resize_fig(ax, ax_size, fig_width, fig_height))

        if "area" in vm.vertexForces and not("volume" in vm.vertexForces):
            cbar_area = fig.colorbar(
                mappable=scalarMap_yketa, ax=ax,
                shrink=0.75, pad=0.01)
            cbar_area.set_label(
                r"$(A_i - A_0)/A_0$",
                rotation=270, labelpad=_cbar_labelpad)
            # resize
            ax_size, fig_width, fig_height = (
                _resize_fig(ax, ax_size, fig_width, fig_height))

        if "surface" in vm.vertexForces:
            assert(not("volume" in vm.vertexForces))
            assert(not("area" in vm.vertexForces))
            assert(not("perimeter" in vm.vertexForces))
            cbar_area = fig.colorbar(
                mappable=scalarMap_yketa, ax=ax,
                shrink=0.75, pad=0.01)
            if cbar_zero == 'hexagon':
                cbar_area.set_label(
                    r"$(h_i - h_6^*)/h_6^*$",
                    rotation=270, labelpad=_cbar_labelpad)
                
            elif cbar_zero == 'average':
                cbar_area.set_label(
                    r"$(h_i - \bar{h})/\bar{h}$",
                    rotation=270, labelpad=_cbar_labelpad)
            elif cbar_zero == 'absolute':
                cbar_area.set_label(
                    r"$h_i / r_6^*$",
                    rotation=270, labelpad=_cbar_labelpad)
            if surface_force.parameters["tauV"] == 0 or True:
                pass
            else:
                cbar_area.set_label(
                    r"$(V_i - V_0)/V_0$",
                    rotation=270, labelpad=_cbar_labelpad)
            # resize
            ax_size, fig_width, fig_height = (
                _resize_fig(ax, ax_size, fig_width, fig_height))

        if "out" in vm.halfEdgeForces:
            cbar_tension = fig.colorbar(
                mappable=scalarMap_yketa, ax=ax,
                shrink=0.75, pad=0.01)
            cbar_tension.set_label(
                r"$(t_i - \left<t_i\right>)/\mathrm{std}(t_i)$",
                rotation=270, labelpad=_cbar_labelpad)
            # resize
            ax_size, fig_width, fig_height = (
                _resize_fig(ax, ax_size, fig_width, fig_height))

        for model in ["model%i" % i for i in range(5)]:
            if model in vm.halfEdgeForces:
                cbar_tension = fig.colorbar(
                    mappable=scalarMap_yketa, ax=ax,
                    shrink=0.75, pad=0.01)
                cbar_tension.set_label(
                    r"$(t_i - \left<t_i\right>)/\mathrm{std}(t_i)$",
                    rotation=270, labelpad=_cbar_labelpad)
                # resize
                ax_size, fig_width, fig_height = (
                    _resize_fig(ax, ax_size, fig_width, fig_height))


        # set figure limits
        _set_lim(ax, vm)
        fig.canvas.mpl_connect("button_press_event",    # reset figure limits on double click
            lambda event: event.dblclick and _set_lim(ax, vm))



    # PLOT

    plt.sca(ax)
    try:
        # make zoom persistent
        # https://discourse.matplotlib.org/t/how-to-make-zoom-persistent/22663/3
        fig.canvas.toolbar.push_current()
        ax.cla()
        fig.canvas.toolbar.back()
    except AttributeError:
        ax.cla()
        _set_lim(ax, vm)
    if only_set: return fig, ax



    # junctions and half-edges
    lines = LineCollection(getLinesJunction(vm), colors="pink",                 # all junctions
        linewidths=2.5/max(1, np.sqrt(len(vm.vertices))/12))                    # scale junction width with linear system size
    if ("t0" in locals() or "sT0" in locals()):
        junctions = vm.getHalfEdgeIndicesByType("junction")
        if "t0" in locals():
            tensions = (
                lambda tension:
                    np.concatenate(list(map(
                        lambda i: [tension[i]]*2,
                        junctions))))(
                vm.halfEdgeForces["out"].tension)
        elif "sT0" in locals():
            for model in ["model%i" % i for i in range(5)]:
                try:
                    tensions = (
                        lambda tension: np.concatenate(list(map(
                            lambda i: [tension[i]]*2,
                            junctions))))(
                        vm.halfEdgeForces[model].tension)
                except:
                    continue
        tensions_std = tensions.std()
        if tensions_std != 0:
            lines.set_color(list(map(
                lambda s_tension: scalarMap_yketa.to_rgba(s_tension),
                (tensions - tensions.mean())/tensions_std)))
    ax.add_collection(lines)

    # cells
    polygons = PatchCollection(
        list(map(                                   # all cells
            lambda vertices: plt.Polygon(vertices, closed=True),
            getPolygonsCell(vm))),
        facecolors="none", edgecolors="none")
    cells = vm.getVertexIndicesByType("centre")

    assert cbar_zero == 'hexagon' or cbar_zero == 'average' or cbar_zero == 'absolute'

    if cbar_zero == 'hexagon':

        if "volume_force" in locals():
            s_val = (                           # colour according to height
                lambda height: list(map(
                    lambda i: (height[i] - h0)/h0,  # silja: what is height?
                    cells)))(
                heights)
        elif "surface_force" in locals():
            if surface_force.parameters["tauV"] == 0 or True:
                s_val = list(map(               # colour according to height
                    lambda i: (heights[i] - h6[i])/h6[i],
                    cells))
            else:
                s_val = list(map(
                    lambda i: (volume[i] - surface_force.parameters["volume"]
                        )/surface_force.parameters["volume"],
                    cells))
        elif "A0" in locals():
            areas = 0                           # silja: missing, must be updated
            s_val = (areas - A0)/A0             # colour according to area


    if cbar_zero == 'average':
        assert "surface_force" in locals()
        assert surface_force.parameters["tauV"] == 0 or True
        
        s_val = list(map(               # colour according to height
            lambda i: (heights[i] - np.mean(h)) / np.mean(h),
            cells))
        
    if cbar_zero == 'absolute':
        assert "surface_force" in locals()
        assert surface_force.parameters["tauV"] == 0 or True
        
        s_val = list(map(               # colour according to height
            lambda i: heights[i],
            cells))
 

    polygons.set_facecolor(list(map(
        lambda v: scalarMap_yketa.to_rgba(v),
        s_val)))
    ax.add_collection(polygons)



    # vertex indices
    if vertex_indices:
        (lambda vertices: list(map(
            lambda i: ax.text(*vertices[i].position, i),
            vertices)))(
        vm.vertices)

    title = (r"$t=%.3f, N_{\mathrm{T}_1}=%.3e, N_{\mathrm{cells}}=%i$"
        % (vm.time, vm.nT1, len(cells)))
    if "A0" in locals():
        title += r"$, A_0=%.1e$" % A0
    if "eta" in locals():
        title += r"$, \eta=%.1e$" % eta
    if "gamma" in locals():
        title += r"$, \gamma=%.1e$" % gamma
    if "p0" in locals():
        title += r"$, p_0=%.2f$" % p0
    if "P0" in locals():
        title += r"$, P_0=%.2f$" % P0
    if "v0" in locals():
        title += r"$, v_0=%.1e, \tau_p=%.1e$" % (v0, taup)
    if "t0" in locals():
        title += r"$, t_0=%1.e, \tau_p=%.1e$" % (t0, taup)
    if "sT0" in locals():
        title += r"$, \sigma=%.1e, \tau_p=%.1e$" % (sT0, taup)
    if "taur" in locals():
        title += r"$, \tau_r=%.1e$" % taur
    ax.set_title(title)

    # update canvas
    if update: _update_canvas(fig)

    return fig, ax


# COLOURBARS

# resize figure with colourbars
# (https://github.com/matplotlib/matplotlib/issues/15010#issuecomment-524438047)
def _measure_fig(ax):
    ax_size = ax.get_position().size.copy()
    try:
        fig_width, fig_height = (
            lambda s: (s.width(), s.height()))(
            ax.figure.canvas.window().size())
    except AttributeError:
        fig_width, fig_height = None, None
    return ax_size, fig_width, fig_height

def _resize_fig(ax, ax_size, fig_width, fig_height):
    r = ax_size/ax.get_position().size  # rescaling factors
    # rescale
    ax.figure.set_size_inches(ax.figure.get_size_inches()*r)
    try:
        ax.figure.canvas.window().setFixedSize(
            int(fig_width*r.max()), fig_height)
    except AttributeError: pass
    # measure again
    return _measure_fig(ax)

_cbar_labelpad = 40 # label spacing from axis

# area colourbar
cmap_yketa = (                                                               # colourmap
    LinearSegmentedColormap.from_list("aroace", (                           # https://en.wikipedia.org/wiki/Pride_flag#/media/File:Aroace_flag.svg
        (0/4, (0.125, 0.220, 0.337)),
        (1/4, (0.384, 0.682, 0.863)),
        (2/4, (1.000, 1.000, 1.000)),
        (3/4, (0.925, 0.804, 0.000)),
        (4/4, (0.886, 0.549, 0.000))))
    or plt.cm.bwr)
norm_area = Normalize(-1, 1)                                                # interval of value represented by colourmap
scalarMap_yketa = ScalarMappable(norm_area, cmap_yketa)                       # conversion from scalar value to colour
