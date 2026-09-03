import numpy as np

def dataset_mean(arr):
    dims = np.shape(arr)

    if len(dims) == 3:
        mean = np.ma.mean(arr, axis=(0,2))
    else:
        mean = np.ma.mean(arr)

    return mean


def dataset_std(arr):
    dims = np.shape(arr)

    if len(dims) == 3:
        std = np.ma.std(arr, axis=(0,2))
    else:
        std = np.ma.std(arr)

    return std


def timepoint_mean(arr):
    arr  = np.ma.masked_invalid(arr)
    dims = np.shape(arr)

    if len(dims) == 3:
        mean = np.ma.mean(arr, axis=(1,2), keepdims=True)
    else:
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

    return 0.5*(x[1:] + x[:-1]), y, bins


def rescaled_distribution(arr, bins=None, hist_range=None):
    arr = np.ma.masked_invalid(arr)
    if arr.count() == 0:
        return np.array([]), np.array([])

    tmean = timepoint_mean(arr)
    tmean = np.ma.masked_invalid(tmean)
    tmean = np.ma.masked_where(tmean == 0, tmean)

    arr_rescaled = arr / tmean
    arr_rescaled = np.ma.masked_invalid(arr_rescaled)

    arr_flat = arr_rescaled.compressed()
    if arr_flat.size == 0:
        return np.array([]), np.array([])
    
    obs_rescaled, freq, _ = hist_to_curve(arr_flat, bins=bins, hist_range=hist_range)

    return obs_rescaled, freq