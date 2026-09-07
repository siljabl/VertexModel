import pickle
import numpy as np

from operator import itemgetter
from paths_config import SIM_RAW_DIR, FRAME_TO_H
from vm_observables import *
from exp_observables import get_attribute, pad_datasets_to_common_shape
from matrix_preprocessing import pad_and_stack_2d, pad_and_stack_axis, interp_corr2d_to_t_common


def simulation_observable(vms, func):
    """
    vms  : list of runs, each run is list_vm (frames)
    func : function(list_vm) -> array (T, N)

    Returns
    -------
    obs : np.ma.array
        Shape (T, R, N): frame, run, cell.
    """

    obs = []

    for vm in vms:
        observable = func(vm)
        obs.append(observable)

    obs = np.ma.array(obs)

    return np.swapaxes(obs, 0, 1)



def experiment_observable(exps_at_density, attr):
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

    obs = np.ma.array(obs)
    obs = np.ma.swapaxes(obs, 0, 1)

    return obs




def spatial_correlation(corr_list, parameter, r_ref=None):
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



def temporal_correlation(corr_list, parameter, mean_var='r', t_ref=None):
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
            N_rho, N_t = np.shape(C_t)

            # target number of lags when resampled to common dt
            N_t_common = int(N_t / f2h)
            t_interp = np.arange(N_t_common) * min(FRAME_TO_H)
            C_t = interp_corr2d_to_t_common(C_t, f2h, t_interp)

        per_corr.append(C_t)

    # Pad along time (last axis) and stack over objects
    C_t_ensemble = pad_and_stack_2d(per_corr)  # shape (N_obj, N_rho, N_t_max)

    # Define a common t_array (indices 0..N_t_max-1)
    N_t_max = C_t_ensemble.shape[-1]
    t_ref = np.arange(N_t_max) * min(FRAME_TO_H)



    return C_t_ensemble, t_ref
