import numpy as np


def rescaled_population_mean(arr):
    dims = np.shape(arr)

    if len(dims) == 3:
        mean = np.ma.mean(arr, axis=(1,2), keepdims=True)
    else:
        mean = np.ma.mean(arr, axis=1, keepdims=True)

    return arr / mean


def hist_to_curve(arr, bins=None, hist_range=None):
    ''' Returns histogram as a normalized curve '''
    if hist_range == None:
        hist_range = (np.ma.min(arr), np.ma.max(arr))

    if bins == None:
        bins  = int(np.max(arr))

    y, x = np.histogram(arr, bins=bins, range=hist_range, density=True)

    return 0.5*(x[1:] + x[:-1]), y, bins


def rescaled_distribution(arr, bins=None):

    arr_rescaled = rescaled_population_mean(arr)
    arr_rescaled = arr_rescaled.flatten()
    obs_rescaled, freq, _ = hist_to_curve(arr_rescaled, bins=bins)

    return obs_rescaled, freq