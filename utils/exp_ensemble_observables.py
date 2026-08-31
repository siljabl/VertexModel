import pickle
import numpy as np

from operator import itemgetter
from cells.bind import VertexModel, getPolygonsCell
from paths_config import SIM_RAW_DIR, FRAME_TO_H


def pad_datasets_to_common_shape(datasets):
    """
    Pad all masked-array attributes of each dataset so they share the same shape
    (T_max, N_max). Extra rows/columns are fully masked.

    Parameters
    ----------
    datasets : list
        List of dataset objects, each with attributes:
        x, y, h, A, dx, dy, label, aminor, amajor, theta, ecc, density (and optionally n),
        where per-cell arrays have shape (T, N).

    Returns
    -------
    datasets : list
        The same list, modified in place so that all per-cell arrays are (T_max, N_max),
        dx/dy are (T_max - 1, N_max), and density is (T_max,).
    """
    # 1. Find max number of frames and cells over all datasets
    T_max = 0
    N_max = 0
    for ds in datasets:
        T, N = ds.A.shape  # use area as reference
        T_max = max(T_max, T)
        N_max = max(N_max, N)

    for ds in datasets:
        T, N = ds.A.shape

        # --- pad per-cell arrays to (T_max, N_max) ---
        def pad_2d(arr, T, N, T_max, N_max):
            """Pad 2D masked array arr (T, N) to (T_max, N_max)."""
            padded = np.ma.masked_all((T_max, N_max), dtype=arr.dtype)
            padded[:T, :N] = arr
            return padded

        ds.x      = pad_2d(ds.x,      T, N, T_max, N_max)
        ds.y      = pad_2d(ds.y,      T, N, T_max, N_max)
        ds.h      = pad_2d(ds.h,      T, N, T_max, N_max)
        ds.A      = pad_2d(ds.A,      T, N, T_max, N_max)
        ds.label  = pad_2d(ds.label,  T, N, T_max, N_max)
        ds.aminor = pad_2d(ds.aminor, T, N, T_max, N_max)
        ds.amajor = pad_2d(ds.amajor, T, N, T_max, N_max)
        ds.theta  = pad_2d(ds.theta,  T, N, T_max, N_max)
        ds.ecc    = pad_2d(ds.ecc,    T, N, T_max, N_max)

        # if hasattr(ds, "n"):
        #     ds.n = pad_2d(ds.n, T, N, T_max, N_max)

        # --- recompute dx, dy from padded x, y ---
        ds.dx = np.ma.diff(ds.x, axis=0)  # shape (T_max - 1, N_max)
        ds.dy = np.ma.diff(ds.y, axis=0)

        # --- recompute density from padded A ---
        ds.density = 10 ** 6 / np.ma.mean(ds.A, axis=1)  # shape (T_max,)

    return datasets


def get_attribute(exp, attr, f2h):
    attrs = dict(vars(exp))

    if attr == "V":
        h = attrs["h"]
        A = attrs["A"]
        return h*A

    elif attr == "pxy":
        amin = attrs["aminor"]
        amaj = attrs["amajor"]
        return amaj / amin
    
    elif attr == "ph":
        h = attrs["h"]
        A = attrs["A"]
        return h / np.ma.sqrt(A)
    
    elif attr == "v":
        dx = attrs["dx"]
        dy = attrs["dy"]
        return np.ma.sqrt(dx**2 + dy**2) / f2h
    
    else: 
        assert hasattr(exp, attr), f"Datasets do not have attribute '{attr}'."
        arr = attrs[attr]

    return arr


def ensemble_observable(exps_at_density, attr):
    """
    vms  : list of runs, each run is list_vm (frames)
    func : function(list_vm) -> array (T, N)

    Returns
    -------
    obs : np.ma.array
        Shape (T, R, N): frame, run, cell.
    """

    #assert hasattr(exps_at_density[0], attr), f"Datasets do not have attribute '{attr}'."

    obs = []
    exps = pad_datasets_to_common_shape(exps_at_density)

    for exp, f2h in zip(exps, FRAME_TO_H):

        attr_arr = get_attribute(exp, attr, f2h)
        obs.append(attr_arr)

    return np.ma.swapaxes(obs, 0, 1)
