import numpy as np


def hist_to_curve(arr, bins=0, hist_range=None):
    ''' Returns histogram as a normalized curve '''
    if hist_range == None:
        hist_range = (np.ma.min(arr), np.ma.max(arr))

    if bins == 0:
        bins  = int(np.max(arr))

    y, x = np.histogram(arr, bins=bins, range=hist_range, density=True)

    return 0.5*(x[1:] + x[:-1]), y, bins
