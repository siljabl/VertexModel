import numpy as np
from scipy.stats import linregress



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
