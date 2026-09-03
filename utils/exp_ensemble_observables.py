import pickle
import numpy as np

from operator import itemgetter
from cells.bind import VertexModel, getPolygonsCell
from paths_config import SIM_RAW_DIR, FRAME_TO_H


def pad_2d(arr, T, N, T_max, N_max):
    """Pad 2D masked array arr (T, N) to (T_max, N_max)."""
    padded = np.ma.masked_all((T_max, N_max), dtype=arr.dtype)
    padded[:T, :N] = arr

    return padded



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
        h = np.ma.clip(attrs["h"], 0, 1e4) 
        A = np.ma.clip(attrs["A"], 0, 1e4)
        return np.ma.masked_invalid(h*A)

    elif attr == "pxy":
        amin = np.ma.clip(attrs["aminor"], 1e-8, 1e4)
        amaj = np.ma.clip(attrs["amajor"], 0,    1e4)
        return np.ma.masked_invalid(amaj / amin)
    
    elif attr == "ph":
        h = np.ma.clip(attrs["h"], 0, 1e4)
        A = np.ma.clip(attrs["A"], 1, 1e4)
        return np.ma.masked_invalid(h / np.ma.sqrt(A))
    
    elif attr == "v":
        dx = np.ma.masked_invalid(attrs["dx"])
        dy = np.ma.masked_invalid(attrs["dy"])
        return np.ma.masked_invalid(np.ma.sqrt(dx**2 + dy**2) / f2h)
    
    else: 
        assert hasattr(exp, attr), f"Datasets do not have attribute '{attr}'."
        arr = np.ma.masked_invalid(attrs[attr])

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




def pad_and_stack_axis(arrays, axis=-1):
    """
    Pad a list of arrays along a given axis to the same length and stack.

    Parameters
    ----------
    arrays : list[np.ndarray or np.ma.MaskedArray]
        All arrays must have the same shape except along `axis`.
    axis : int
        Axis along which to pad (can be negative).

    Returns
    -------
    stacked : np.ma.MaskedArray
        Shape (N, *batch_shape, L_max) after moving `axis` to the end,
        where:
        - N          : number of arrays,
        - batch_shape: all axes except `axis`,
        - L_max      : max length along `axis`.

    """
    arrays = [np.ma.array(a) for a in arrays]

    # Normalize axis (handle negative indices)
    ndim = arrays[0].ndim
    axis_norm = axis if axis >= 0 else ndim + axis

    # Move target axis to the end for all arrays
    moved = [np.moveaxis(a, axis_norm, -1) for a in arrays]

    # Find max length along that axis
    L_max = max(a.shape[-1] for a in moved)
    batch_shape = moved[0].shape[:-1]

    padded_list = []
    for a in moved:
        padded = np.ma.masked_all(batch_shape + (L_max,), dtype=a.dtype)
        padded[..., :a.shape[-1]] = a
        padded_list.append(padded)

    # Stack over the new leading axis (arrays index)
    stacked = np.ma.stack(padded_list, axis=0)  # (N, *batch_shape, L_max)
    stacked = np.moveaxis(stacked, -1, axis+1)

    return stacked


def pad_and_stack_2d(arrays):
    """
    Pad a list of 2D arrays to the same shape and stack them.

    Parameters
    ----------
    arrays : list[np.ndarray or np.ma.MaskedArray]
        Each array has shape (n_i, m_i).

    Returns
    -------
    stacked : np.ma.MaskedArray
        Masked array of shape (N, n_max, m_max), where:
        - N      : number of arrays
        - n_max  : max number of rows across arrays
        - m_max  : max number of columns across arrays
        Extra entries are fully masked.
    """
    arrays = [np.ma.array(a) for a in arrays]

    # Find max dimensions
    n_max = max(a.shape[0] for a in arrays)
    m_max = max(a.shape[1] for a in arrays)

    padded_list = []
    for a in arrays:
        n, m = a.shape
        padded = np.ma.masked_all((n_max, m_max), dtype=a.dtype)
        padded[:n, :m] = a
        padded_list.append(padded)

    stacked = np.ma.stack(padded_list, axis=0)  # (N, n_max, m_max)
    return stacked



def _interp_corr2d_to_t_common(C_t2d, dt, t_common):
    """
    Interpolate a 2D correlation array (N_density, N_t) onto t_common along the time axis.

    Parameters
    ----------
    C_t2d : 2D array
        Correlations at lags k*dt, shape (N_density, N_t).
    dt : float
    t_common : 1D array

    Returns
    -------
    C_interp : 2D array
        Shape (N_density, len(t_common)).
    """
    C_t2d = np.asarray(C_t2d)
    N_density, N_t = C_t2d.shape
    t_source = dt * np.arange(N_t)

    C_interp = np.empty((N_density, t_common.shape[0]), dtype=C_t2d.dtype)
    for i in range(N_density):
        C_interp[i] = np.interp(t_common, t_source, C_t2d[i])
    return C_interp




def ensemble_spatial_correlation(corr_list, parameter, r_ref=None):
    """
    Build an ensemble of spatial correlations C(r) across multiple Autocorrelations objects.

    Parameters
    ----------
    corr_list : list[Autocorrelations]
        Each object must have corr.spatial[parameter] defined.
    parameter : str
        Observable identifier, e.g. 'hh', 'AA', 'VV', 'vv'.

    Returns
    -------
    C_r_ensemble : np.ma.array
        Masked array of shape (N_r, N_ensemble, n_comp), where:
        - N_r        : number of r-lags
        - N_ensemble : number of elements in corr_list
        - n_comp     : correlation components per r (typically 1 if just C(r))
    r_array : np.ndarray
        1D array of r-lag centers of shape (N_r,), from the first element.
    """
    per_corr = []

    for corr in corr_list:
        C_r = corr.spatial[parameter]        # shape (N_r, n_comp)
        r   = corr.r_array[parameter]        # shape (N_r,)

        if r_ref is None:
            r_ref = r
        else:
            if len(r) != len(r_ref) or not np.allclose(r_ref, r):
                raise ValueError("r_array mismatch between correlation objects")

        per_corr.append(C_r)

    # Stack: (N_ensemble, N_rho, N_r)
    C_r_ensemble = pad_and_stack_axis(per_corr, 0)

    return C_r_ensemble, r_ref



def ensemble_temporal_correlation(corr_list, parameter, mean_var='r', t_ref=None):
    """
    Build an ensemble of temporal correlations C(t) across multiple Autocorrelations objects.

    Parameters
    ----------
    corr_list : list[Autocorrelations]
        Each object must have either corr.temporal[parameter] or corr.temporal_cell[parameter],
        depending on mean_var.
    parameter : str
        Observable identifier.
    mean_var : {'r', 'cell'}
        'r'      : use corr.temporal[parameter]
        'cell'   : use corr.temporal_cell[parameter]

    Returns
    -------
    C_t_ensemble : np.ma.array
        Masked array of shape (N_t, N_ensemble), where:
        - N_t        : number of time lags
        - N_ensemble : number of elements in corr_list
    t_array : np.ndarray
        1D array of time lags of shape (N_t,), from the first element.
    """
    assert mean_var in ('r', 'cell')

    per_corr = []

    for corr, f2h in zip(corr_list, FRAME_TO_H):
        
        if mean_var == 'r':
            C_t = corr.temporal[parameter]       # shape (N_rho, N_t)
        else:
            C_t = corr.temporal_cell[parameter]  # shape (N_rho, N_t)

        if f2h > min(FRAME_TO_H) and len(C_t) > 0:
            t_interp = np.arange(0, len(C_t), min(FRAME_TO_H))
            C_t = _interp_corr2d_to_t_common(C_t, f2h, t_interp)

        per_corr.append(C_t)

    # Pad along time (last axis) and stack over objects
    C_t_ensemble = pad_and_stack_2d(per_corr)  # shape (N_obj, ..., N_t_max)

    # Define a common t_array (indices 0..N_t_max-1)
    N_t_max = C_t_ensemble.shape[-1]
    t_ref = np.arange(N_t_max)



    return C_t_ensemble, t_ref * min(FRAME_TO_H)
