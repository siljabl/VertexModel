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
from scipy.optimize import curve_fit
from paths_config import SIM_RAW_DIR


def exp_decay(lags, L_eff):
    return np.exp(-lags / L_eff) 


import numpy as np

def smooth_1d(data, window, mode="block"):
    """
    Smooth a 1D array or MaskedArray in time.

    Parameters
    ----------
    data : 1D np.ndarray or np.ma.MaskedArray
        Input data (e.g. C_mean(t)).
    window : int
        Number of points per smoothing window (e.g. 3).
    mode : {"block", "moving"}
        "block"   : non-overlapping block average (like script 2).
        "moving"  : simple moving average (convolution).

    Returns
    -------
    smoothed : np.ma.MaskedArray
        Smoothed data.
    indices  : np.ndarray
        Indices in the original array corresponding to the smoothed points.
        Use these to index your time/lag array.
    """
    data = np.ma.array(data)

    if mode == "block":
        n_blocks = data.shape[0] // window
        if n_blocks == 0:
            return data, np.arange(data.shape[0])

        trimmed = data[:n_blocks * window]
        smoothed = trimmed.reshape(-1, window).mean(axis=1)

        # take the mean index of each block as representative
        idx = np.arange(n_blocks * window).reshape(-1, window).mean(axis=1).astype(int)
        return smoothed, idx

    elif mode == "moving":
        # simple moving average; keep only valid points
        kernel = np.ones(window) / window
        smoothed = np.ma.convolve(data, kernel, mode="valid")

        # center the kernel; indices for smoothed points
        offset = (window - 1) // 2
        idx = np.arange(offset, offset + smoothed.shape[0])
        return smoothed, idx

    else:
        raise ValueError("mode must be 'block' or 'moving'")



def symmetric_spacing(r):
    r = np.asarray(r)
    dr = np.empty_like(r, dtype=float)

    # interior points: average of left and right spacing
    dr[1:-1] = (r[2:] - r[:-2]) / 2.0

    # boundaries: use one-sided spacing
    dr[0]  = r[1]  - r[0]
    dr[-1] = r[-1] - r[-2]

    return dr


def find_lag0(lags, C_ensemble, threshold):
    """
    Find first lag where C crosses below `threshold` for each dataset.

    Parameters
    ----------
    lags : 1D array
        Lag values.
    C_ensemble : 2D array or MaskedArray
        Shape (N_datasets, N_lags).
    threshold : float

    Returns
    -------
    lags_0 : np.ma.MaskedArray
        Shape (N_datasets,), masked where no valid crossing / data.
    """

    C_ds_mean = np.ma.mean(C_ensemble, axis=1)
    N_ds = C_ds_mean.shape[0]

    # Preallocate masked array: everything masked initially
    lags_0 = np.ma.masked_all(N_ds, dtype=float)

    for C, i in zip(C_ds_mean, range(N_ds)):

        # Indices where C < threshold
        idx = np.ma.where(C < threshold)[0]

        if len(idx) > 0 :
            lag_0 = lags[idx[0]]
        elif len(idx) == 0 and np.ma.sum(C) > 0:
            # Never drops below threshold but stays positive: use last lag
            lag_0 = lags[-1]
        else:
            # No data / nonpositive everywhere
            continue

        lags_0[i] = lag_0 

    return lags_0


def integrate_lag(lags, C_ensemble, lags_0, dlag):
    C_ds_mean = np.ma.mean(C_ensemble, axis=1)
    N_ds = C_ds_mean.shape[0]

    # Preallocate masked array: everything masked initially
    C_int = np.ma.masked_all(N_ds, dtype=float)

    i = 0
    for C, lag0 in zip(C_ds_mean, lags_0):
        if np.ma.sum(C) > 0:
            C_int[i] = np.ma.sum((C * dlag)[lags < lag0])
        else:
            continue
        i += 1

    return C_int



def compute_correlation_length(lags, C, threshold=0, method="sum", smooth=False):
    """
    Estimate an effective correlation length/time as the first lag where
    C(lag) drops below `threshold`. If it never does, return max lag.

    Parameters
    ----------
    lags : 1D array
    C : 1D array
    threshold : float

    Returns
    -------
    L_eff : float
    """
    assert method in ("sum", "first", "fit")

    C = np.ma.array(C)

    lag_0 = find_lag0(lags, C, threshold)
    
    if method == "first":

        L_eff = np.ma.mean(lag_0)
        err   = np.ma.std(lag_0)
        # L_eff = err = 1


    elif method == "sum":

        dlag  = symmetric_spacing(lags)
        C_int = integrate_lag(lags, C, lag_0, dlag)

        L_eff = np.ma.mean(C_int)
        err   = np.ma.std(C_int)

        # L_eff = err = 1


    elif method == "fit":
        # Compute correlation length based on exponential fit 
        C_mean = np.ma.mean(C, axis=(0,1))
        if smooth:    
            valid = ~C_mean.mask
            C_mean_valid = C_mean[valid]
            lags_valid   = lags[valid]

            # Smooth (block average with window=3, like script 2)
            C_smooth, idx = smooth_1d(C_mean_valid, window=3, mode="block")
            lags_smooth   = lags_valid[idx]
        try:
            L_eff, err = curve_fit(exp_decay, 
                                lags[(C_mean > threshold)*(lags >= 0)], 
                                C_mean[(C_mean > threshold)*(lags >= 0)])
        except:
            L_eff, err = None
        # L_eff = err = 1


    return L_eff, err


def plot_correlations_by_density(ax_arr, corr_by_density, densities, parameter,
                                 var='r', mean_var='r', cmap='viridis', threshold=0.2,
                                 method="sum", smooth=False):
    """
    Plot correlation functions at different densities and derived quantities.

    Parameters
    ----------
    ax_arr : list[Axes]
        [0] C(lag) vs lag
        [1] C(lag0) vs density (lag0 = 0 or first nonzero)
        [2] correlation length/time vs density
    corr_by_density : list[list[Autocorrelations]]
        corr_by_density[i] = list of Autocorrelations at density densities[i].
    densities : list[float]
        Density values corresponding to corr_by_density.
    parameter : str
        'hh', 'AA', 'VV', 'vv'
    var : {'r', 't'}
        'r' → spatial C(r), 't' → temporal C(t)
    mean_var : {'r', 'cell'}
        For temporal correlations only.
    cmap : str
        Colormap name.
    threshold : float
        Threshold for effective correlation length/time.
    """
    densities = np.array(densities)

    colors = mpl.colormaps[cmap](np.linspace(0.2, 1, len(densities)))
    norm   = mpl.colors.Normalize(vmin=densities.min(), vmax=densities.max())
    sm     = mpl.cm.ScalarMappable(cmap=mpl.colormaps[cmap], norm=norm)

    for corr_list, rho, c in zip(corr_by_density, 
                                 densities, 
                                 colors):
        
        if len(corr_list) == 0:
            continue

        if var == 'r':
            C_ensemble, lags = get_exp.ensemble_spatial_correlation(corr_list, parameter)
        else:
            C_ensemble, lags = get_exp.ensemble_temporal_correlation(corr_list, parameter, mean_var=mean_var)

        # Average correlation across objects
        # C_mean = np.ma.mean(C_ensemble, axis=(0,1))  # (N_lag,)
        C_mean = np.ma.mean(np.ma.mean(C_ensemble, axis=1), axis=0)  # (N_lag,)

        if smooth:
            # Optional: restrict to non-masked points before smoothing
            valid = ~C_mean.mask
            C_mean_valid = C_mean[valid]
            lags_valid   = lags[valid]

            # Smooth (block average with window=3, like script 2)
            C_smooth, idx = smooth_1d(C_mean_valid, window=3, mode="block")
            lags_smooth   = lags_valid[idx]

            # Plot smoothed curve
            ax_arr[0].plot(lags_smooth, C_smooth, c=c)

        else:
            # Panel 0: C(lag)
            ax_arr[0].plot(lags, C_mean, c=c)

        # Panel 1: C(lag0) vs density (here lag0 = 0)
        try:
            L_eff, err = compute_correlation_length(lags, C_ensemble, threshold=threshold, method=method, smooth=smooth)
            ax_arr[1].errorbar(rho, L_eff, err, fmt='o', c=c)
        except:
            None
    plt.colorbar(sm, ax=ax_arr[0], label=r"$\rho_{\text{cell}}$ (mm$^{-2}$)")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Plot correlations by density for simulations")
    parser.add_argument("config_dirpath",  type=str,
                        help="Path to ensemble config. Typically 'configs/mode/params/'")
    parser.add_argument("param",           type=str,
                        help="Correlation parameter ('hh', 'AA', 'VV', 'vv')")
    parser.add_argument("var",             type=str,
                        help="Correlation variable: 'r' (spatial) or 't' (temporal)")
    parser.add_argument("-N", "--Ngrids",  nargs='*',
                        help="Ngrids to loop over", default=[36, 42, 48, 54])
    parser.add_argument("--mean_var",      type=str,
                        help="For temporal correlations: fluctuations around 'r' or 'cell'",
                        default="r")
    parser.add_argument("-c", "--cmap",    type=str,
                        help="Name of colormap", default="viridis")
    parser.add_argument("--ylog",          action="store_true",
                        help="Log scale on correlation plot")
    parser.add_argument("--threshold",     type=float,
                        help="Threshold for effective correlation length/time", default=0.2)

    args = parser.parse_args()

    assert args.var in ("r", "t"), "var must be 'r' or 't'"
    assert args.param in ("hh", "AA", "VV", "vv"), "Unknown param"

    # Set path
    mode    = Path(args.config_dirpath).parent.name
    params  = Path(args.config_dirpath).name
    dirpath = Path(SIM_RAW_DIR) / mode / params

    # Convert to ints
    Ngrids = [int(N) for N in args.Ngrids]

    # Load correlations for each density (Ngrid)
    corr_by_density = []
    for Ngrid in Ngrids:
        corr_list = load_autocorrelations_for_sim_ensemble(dirpath, Ngrid)
        corr_by_density.append(corr_list)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(11, 3))

    if args.var == "r":
        ax[0].set(xlabel=r"$r$", ylabel=rf"$C_{{{args.param}}}(r)$")
        ax[1].set(xlabel=r"$\rho_{{\text{cell}}}$ (1/mm$^2$)", ylabel=rf"$C_{{{args.param}}}(0)$")
        ax[2].set(xlabel=r"$\rho_{{\text{cell}}}$ (1/mm$^2$)",
                  ylabel=rf"$\xi_{{{args.param}}}$ (effective corr. length)")
    else:
        ax[0].set(xlabel=r"$t$", ylabel=rf"$C_{{{args.param}}}(t)$")
        ax[1].set(xlabel=r"$\rho_{{\text{cell}}}$ (1/mm$^2$)", ylabel=rf"$C_{{{args.param}}}(0)$")
        ax[2].set(xlabel=r"$\rho_{{\text{cell}}}$ (1/mm$^2$)",
                  ylabel=rf"$\tau_{{{args.param}}}$ (effective corr. time)")

    plot_correlations_by_density(
        ax_arr=ax,
        corr_by_density=corr_by_density,
        densities=Ngrids,
        parameter=args.param,
        var=args.var,
        mean_var=args.mean_var,
        cmap=args.cmap,
        threshold=args.threshold,
    )

    if args.ylog:
        ax[0].set(yscale="log")

    fig.tight_layout()
    outdir = Path("results/tmp")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{args.param}_C{args.var}_by_density.png", dpi=300)


if __name__ == "__main__":
    main()
