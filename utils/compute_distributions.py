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
    dims = np.shape(arr)

    if len(dims) == 3:
        mean = np.ma.mean(arr, axis=(1,2), keepdims=True)
    else:
        mean = np.ma.mean(arr, axis=1, keepdims=True)

    return  mean


def hist_to_curve(arr, bins=None, hist_range=None):
    ''' Returns histogram as a normalized curve '''
    if hist_range == None:
        hist_range = (np.ma.min(arr), np.ma.max(arr))

    if bins == None:
        bins  = int(np.max(arr))

    y, x = np.histogram(arr, bins=bins, range=hist_range, density=True)

    return 0.5*(x[1:] + x[:-1]), y, bins


def rescaled_distribution(arr, bins=None, hist_range=None):

    arr_rescaled = arr / timepoint_mean(arr)
    arr_rescaled = arr_rescaled.compressed().flatten()
    obs_rescaled, freq, _ = hist_to_curve(arr_rescaled, bins=bins, hist_range=hist_range)

    return obs_rescaled, freq