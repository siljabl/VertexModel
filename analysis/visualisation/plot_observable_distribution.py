import os, sys
from pathlib import Path
sys.path.append("utils")

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import pickle
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import io_functions as io
import vm_observables as get_vm
import exp_ensemble_observables as get_exp
import compute_distributions as compute

from cmcrameri import cm

from paths_config import SIM_RAW_DIR


def plot_mean(ax, x, arr, c, fmt="o"):
    mean = np.ma.mean(arr)
    std  = np.ma.std(arr)

    ax.errorbar(x, mean, std, fmt=fmt, c=c)


def plot_std(ax, x, arr, c, fmt="o"):
    
    mean = np.ma.mean(compute.dataset_std(arr) / compute.dataset_mean(arr))
    std  = np.ma.std(compute.dataset_std(arr) / compute.dataset_mean(arr))

    ax.errorbar(x, mean, std, fmt=fmt, c=c)


def plot_observable_by_density(ax_arr, ds_by_density, densities, func=None, attr=None, 
                               cmap="viridis", bins=20, hist_range=None):
    """
    Plot an observable as a function of cell number density.

    Parameters
    ----------
    ax_arr : list[matplotlib.axes.Axes]
        [0] P(obs) vs obs
        [1] mean(obs) vs density (with std as error bar)
        [2] mean(obs) vs std(obs)/mean(obs)
    vms_density : list
        List of ensembles, one per density.
    Ngrids : list[int]
        Grid sizes corresponding to each ensemble; used to compute cell density ρ.
    func : callable
        Observable extractor: func(list_vm) -> array per run (used by ensemble_observable).
    cmap : str
        Colormap name.
    bins : int
        Number of bins for the distribution.
    """
    assert func != None or attr != None, "Must input either 'func' (vm) or 'attr' (exp)"

    # Compute cell number density in units of cells/mm^2
    densities = np.array(densities)

    # Define colorbar
    colors = mpl.colormaps[cmap](np.linspace(0.4, 0.9, len(densities)))
    norm   = mpl.colors.Normalize(vmin=densities.min(), vmax=densities.max())
    sm     = mpl.cm.ScalarMappable(cmap=mpl.colormaps[cmap], norm=norm)

    # Loop over each ensemble at a given density (ρ) and its color
    for ds_at_density, rho, c in zip(ds_by_density, 
                                     densities,
                                     colors):
        
        if func:
            arr = get_vm.ensemble_observable(ds_at_density, func)
        else:
            arr = get_exp.ensemble_observable(ds_at_density, attr)

        # Ensure masking of any NaN/inf
        arr = np.ma.masked_invalid(arr)

        # Ensure masking of missing cells
        arr = np.ma.masked_where(arr <= 0, arr)

        # Skip if no cells
        if arr.count() == 0:
            continue
            
        obs, freq = compute.rescaled_distribution(arr, bins=bins, hist_range=hist_range)

        ax_arr[0].plot(obs, freq, c=c)
        plot_mean(ax_arr[1], rho, arr, c=c)
        plot_std(ax_arr[2],  rho, arr, c=c)
        # ax_arr[2].plot(mean, std / mean, '.', c=c)

    plt.colorbar(sm, ax=ax_arr[0], label=r"$\rho_{\text{cell}}$ (mm$^{-2}$)")





def main():

    parser = argparse.ArgumentParser(description='Plot data set')
    parser.add_argument('config_dirpath',  type=str,  help="Path to ensemble config. Typically 'configs/mode/params/")
    parser.add_argument('vm_observable',   type=str,  help="Observable")
    parser.add_argument('label',       type=str,  help="Label of observable")
    parser.add_argument('unit',        type=str,  help="Unit of observable")
    parser.add_argument('-N', '--Ngrids',  nargs='*', help="Ngrids to loop over",            default=[36, 42, 48, 54])
    parser.add_argument('-b', '--Nbins',   type=int,  help="Number of bins in distribution", default=32)
    parser.add_argument('-c', '--cmap',    type=str,  help="Name of colormap", default="viridis")
    parser.add_argument('--ylog',          action="store_true", help="rescale data")
    args = parser.parse_args()

    # Assert that observable is a function in vm_observables
    if not hasattr(get_vm, args.vm_observable):
        raise ValueError(
            f"Observable '{args.vm_observable}' is not defined in vm_observables.\n"
            f"Available: {', '.join(sorted(name for name in dir(get_vm) if not name.startswith('_')))}"
        )
    func = getattr(get_vm, args.vm_observable)
    if not callable(func):
        raise TypeError(
            f"vm_observables.{args.vm_observable} exists but is not callable."
        )

    # Set path
    mode    = Path(args.config_dirpath).parent.name
    params  = Path(args.config_dirpath).name
    dirpath = Path(SIM_RAW_DIR) / mode / params

    # Convert to ints
    Ngrids = [int(N) for N in args.Ngrids]


    # Load all ensemble frames 
    vms_density = []
    for Ngrid in Ngrids:
        vms = io.load_simulation_ensemble(dirpath, Ngrid=Ngrid)

        vms_density.append(vms)


    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(11,3))

    ax[0].set(xlabel=rf"${args.label}~/~\langle {args.label} \rangle$",
              ylabel=rf"$P({args.label})$")
    ax[1].set(xlabel=r"$\rho_{{\text{cell}}}$ (1/mm$^2$)",
              ylabel=rf"$\langle {args.label} \rangle$ ({args.unit})")
    ax[2].set(xlabel=rf"$\langle {args.label} \rangle$ ({args.unit})",
              ylabel=rf"$\langle {args.label} \rangle$ ({args.unit})")
            #   ylabel=rf"$\sigma_{{{args.label}}}~/~\langle {args.label} \rangle$")

    plot_observable_by_density(ax, vms_density, Ngrids, func, cmap=args.cmap, bins=args.Nbins)

    if args.ylog:
        ax[0].set(yscale="log")

    fig.tight_layout()
    fig.savefig(f"results/tmp/{args.label}_distribution.png", dpi=300)


if __name__ == "__main__":
    main()

