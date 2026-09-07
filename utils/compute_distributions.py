import numpy as np

def dataset_mean(arr):
    """
    Mean over an entire dataset.

    If arr has shape (T, R, N) (time, run, cell), the mean is taken over
    time and cells, returning shape (R,). Otherwise a global mean is returned.
    """
    dims = np.shape(arr)

    if len(dims) == 3:
        # mean over time (axis 0) and cells (axis 2), keep runs separate
        mean = np.ma.mean(arr, axis=(0,2))
    else:
        # global mean over all entries
        mean = np.ma.mean(arr)

    return mean



def dataset_std(arr):
    """
    Standard deviation over an entire dataset.

    If arr has shape (T, R, N) (time, run, cell), the std is taken over
    time and cells, returning shape (R,). Otherwise a global std is returned.
    """
    dims = np.shape(arr)

    if len(dims) == 3:
        # std over time (axis 0) and cells (axis 2), keep runs separate
        std = np.ma.std(arr, axis=(0,2))
    else:
        # global std over all entries
        std = np.ma.std(arr)

    return std



def timepoint_mean(arr):
    """
    Mean at each timepoint, averaged over runs and cells.

    If arr has shape (T, R, N), returns shape (T, 1, 1) with mean over (R, N).
    If arr has shape (T, N), returns shape (T, 1) with mean over cells.
    """
    arr  = np.ma.masked_invalid(arr)
    dims = np.shape(arr)

    if len(dims) == 3:
        # mean over runs and cells, keep time axis
        mean = np.ma.mean(arr, axis=(1,2), keepdims=True)
    else:
        # mean over "cells" (axis 1), keep time axis
        mean = np.ma.mean(arr, axis=1, keepdims=True)

    return  mean


def hist_to_curve(arr, bins=None, hist_range=None):
    ''' Returns histogram as a normalized curve '''

    arr  = np.ma.masked_invalid(arr)
    if arr.count() == 0:
        return np.array([]), np.array([]), bins

    data = arr.compressed()

    if hist_range == None:
        hist_range = (data.min(), data.max())

    if bins == None:
        bins  = max(10, int(np.ceil(data.max())))

    y, x = np.histogram(data, bins=bins, range=hist_range, density=True)

    # bin centers
    x_mid = 0.5 * (x[1:] + x[:-1])

    return x_mid, y, bins


def rescaled_distribution(arr, bins=None, hist_range=None):
    ''' Distribution of observable rescaled by its timepoint mean. '''

    arr = np.ma.masked_invalid(arr)
    if arr.count() == 0:
        return np.array([]), np.array([])
    
    # Compute timepoint mean
    tmean = timepoint_mean(arr)
    tmean = np.ma.masked_invalid(tmean)
    tmean = np.ma.masked_where(tmean == 0, tmean)

    # Rescale by timepoint mean
    arr_rescaled = arr / tmean
    arr_rescaled = np.ma.masked_invalid(arr_rescaled)

    arr_flat = arr_rescaled.compressed()
    if arr_flat.size == 0:
        return np.array([]), np.array([])
    
    obs_rescaled, freq, _ = hist_to_curve(arr_flat, bins=bins, hist_range=hist_range)

    return obs_rescaled, freq