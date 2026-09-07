import numpy as np

from matrix_preprocessing import pad_2d


def get_attribute(exp, attr, f2h):
    attrs = dict(vars(exp))

    if attr == "V":
        h = np.ma.clip(attrs["h"], 0, 1e4) 
        A = np.ma.clip(attrs["A"], 0, 1e4)
        V = np.ma.masked_invalid(h*A)
        
        return V

    elif attr == "pxy":
        amin = np.ma.clip(attrs["aminor"], 1e-8, 1e4)
        amaj = np.ma.clip(attrs["amajor"], 0,    1e4)
        pxy  = np.ma.masked_invalid(amaj / amin)
        
        return pxy
    
    elif attr == "ph":
        h  = np.ma.clip(attrs["h"], 0, 1e4)
        A  = np.ma.clip(attrs["A"], 1, 1e4)
        ph = np.ma.masked_invalid(h / np.ma.sqrt(A))

        return ph
    
    elif attr == "v":
        dx = np.ma.masked_invalid(attrs["dx"])
        dy = np.ma.masked_invalid(attrs["dy"])
        v  = np.ma.masked_invalid(np.ma.sqrt(dx**2 + dy**2) / f2h)
        
        return v
    
    else: 
        assert hasattr(exp, attr), f"Datasets do not have attribute '{attr}'."
        arr = np.ma.masked_invalid(attrs[attr])
        
        return arr



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