import copy
from vm_observables import cell_density


def simulations_by_density(vms_ensemble, density, bin_size=200, size=600):
    """ Load all experiments in given density range """

    runs_at_density = []

    min_density = density - bin_size / 2
    max_density = density + bin_size / 2

    # Loop through experiments and mask data
    for list_vm in vms_ensemble:

        # Compute cell density during simulation
        densities = cell_density(list_vm)

        mask = (densities > min_density) * (densities < max_density)
        run = []
        for vm, m in zip(list_vm, mask):
            if m:
                run.append(vm)

        runs_at_density.append(run)

    return runs_at_density



def experiments_by_density(full_experiments, density, bin_size=200):
    """ Load all experiments in given density range """

    experiments_at_density = []

    min_density = density - bin_size / 2
    max_density = density + bin_size / 2

    # Loop through experiments and mask data
    for exp in full_experiments:

        exp_by_density = copy.copy(exp)

        mask = (exp.density > min_density) * (exp.density < max_density)

        exp_by_density.filter_by_density(mask)
        exp_by_density.drop_empty_cells()
    
        experiments_at_density.append(exp_by_density)

    return experiments_at_density



def correlations_by_density(full_correlations, density, bin_size=200):
    """
    Select correlations from ExperimentalAutocorrelations objects within a given density bin.

    Parameters
    ----------
    full_correlations : list[ExperimentalAutocorrelations]
        List of ExperimentalAutocorrelations objects.
    density : float
        Target density value (same units as corr.density).
    bin_size : float
        Width of the density bin. The function keeps densities in
        (density - bin_size/2, density + bin_size/2).

    Returns
    -------
    correlations_at_density : list[ExperimentalAutocorrelations]
        New list of ExperimentalAutocorrelations objects, each filtered in density
        (i.e. only correlations for densities in the chosen bin are kept).
    """
    
    correlations_at_density = []

    min_density = density - bin_size / 2
    max_density = density + bin_size / 2

    for corr in full_correlations:

        print("densities in range: ", int(corr.density.min()), int(corr.density.max()))

        # Skip if no density info
        if corr.density is None or len(corr.density) == 0:
            print("One dataset skipped because of missing density data")
            continue

        corr_by_density = copy.deepcopy(corr)

        # Boolean mask over ensemble index
        mask = (corr.density > min_density) & (corr.density < max_density)

        # Filter density itself
        corr_by_density.density = corr.density[mask]

        # Filter spatial correlations: shape (N_ensemble, ...)
        for param, C_r in corr.spatial.items():
            if param == "vv":
                corr_by_density.spatial[param] = C_r[mask[:-1]]
            else:
                corr_by_density.spatial[param] = C_r[mask]

        # Filter temporal correlations with mean_var='r'
        for param, C_t in corr.temporal.items():
            if param == "vv":
                corr_by_density.temporal[param] = C_t[mask[:-1]]
            else:
                corr_by_density.temporal[param] = C_t[mask]

        # Filter temporal correlations with mean_var='cell'
        for param, C_t_cell in corr.temporal_cell.items():
            if param == "vv":
                corr_by_density.temporal_cell[param] = C_t_cell[mask[:-1]]
            else:
                corr_by_density.temporal_cell[param] = C_t_cell[mask]

        # t_array and r_array do not depend on density
        corr_by_density.t_array = corr.t_array.copy()
        corr_by_density.r_array = corr.r_array.copy()

        correlations_at_density.append(corr_by_density)
    
    return correlations_at_density


