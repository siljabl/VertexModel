import numpy as np
from scipy.stats import linregress



def pad_2d(arr, T, N, T_max, N_max):
    """Pad 2D masked array arr (T, N) to (T_max, N_max)."""
    padded = np.ma.masked_all((T_max, N_max), dtype=arr.dtype)
    padded[:T, :N] = arr

    return padded



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

    # Move padded axis back to original position (+1 due to leading N axis)
    stacked = np.moveaxis(stacked, -1, axis_norm + 1)

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



def interp_corr2d_to_t_common(C_t2d, dt, t_common):
    """
    Interpolate a 2D correlation array (N_density, N_t) onto t_common along the time axis,
    respecting masks: for each row, only unmasked points are used for interpolation, and
    interpolated values beyond the last unmasked lag are masked.

    Parameters
    ----------
    C_t2d : 2D array-like
        Correlations at lags k*dt, shape (N_density, N_t). Can be np.ndarray or
        np.ma.MaskedArray.
    dt : float
        Time step of the source data (units of time per frame).
    t_common : 1D array-like
        Target time grid (in the same physical units as dt * frame index).

    Returns
    -------
    C_interp : np.ma.MaskedArray
        Shape (N_density, len(t_common)).
    """
    # Ensure masked array
    C_t2d = np.ma.asarray(C_t2d)
    t_common = np.asarray(t_common)

    N_density, N_t = C_t2d.shape
    C_interp = np.ma.masked_all((N_density, t_common.shape[0]), dtype=C_t2d.dtype)

    for i in range(N_density):
        row = C_t2d[i]  # row is 1D masked array

        # Boolean mask of unmasked entries
        unmasked = ~row.mask

        # If the entire row is masked, leave it fully masked
        if not np.any(unmasked):
            continue

        # Source times and values for interpolation (only unmasked points)
        # Use original frame indices (0..N_t-1) scaled by dt
        frame_indices = np.arange(N_t)
        t_source = dt * frame_indices[unmasked]
        values   = row.data[unmasked]

        # Perform interpolation over t_common using these points
        C_tmp = np.interp(t_common, t_source, values)

        # Build mask:
        # - mask all times beyond the last available unmasked lag
        last_t = t_source[-1]
        mask = t_common > last_t

        # Create masked array for this row
        C_interp[i] = np.ma.array(C_tmp, mask=mask)

    return C_interp



def detrend_array(t_arr, arr, keepdims=False):
    """
    Fit and remove a linear trend from a 1D masked array.
    """

    # Fit only on unmasked entries
    fit = linregress(t_arr[~arr.mask], arr.data[~arr.mask])

    if not keepdims:
        # Detrend only on unmasked points, return 1D array
        lin_fit   = t_arr[~arr.mask] * fit.slope
        detrended = arr.data[~arr.mask] - lin_fit

        # Compute relative standard deviation
        rel_std   = np.ma.std(detrended) / np.ma.mean(arr.data[~arr.mask])

        return detrended, rel_std
    
    else:
        # Detrend full array, keeping shape and mask
        lin_fit   = t_arr * fit.slope + fit.intercept
        detrended = arr - lin_fit + np.ma.mean(arr)

        # Compute relative standard deviation
        rel_std = np.ma.std(detrended) / np.ma.mean(arr)

        return detrended, rel_std
    


def detrend_entire_matrix(arr):
    """
    Detrend each column (cell) in a time × cell matrix.
    """

    # Time axis 0
    time = np.arange(len(arr))
    detrended_arr = []

    # Looping through cells in array
    for cell in arr.T:
        try:
            detrended, _  = detrend_array(time, cell, keepdims=True)
            detrended_arr.append(detrended)
        
        except:
            # If regression fails (e.g. all masked), return fully masked column
            detrended_arr.append(np.ma.array(cell, mask=True))

    detrended_arr = np.ma.array(detrended_arr).T

    return detrended_arr
