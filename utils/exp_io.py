import sys
import copy
from pathlib import Path
# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


from SegmentedCells import SegmentedCells
from Correlations   import ExperimentalAutocorrelations
from paths_config   import EXP_RAW_DIR, EXP_PROC_DIR, EXP_DATASETS_IDS



def load_full_experiments(DATASET_IDS=EXP_DATASETS_IDS):
    """ Load full dataset from all experiments in EXP_DATASETS_IDS """

    experiments = []
    for ID in DATASET_IDS:
        cellprop = SegmentedCells(f"{EXP_RAW_DIR}/{ID}_cells.p")
        experiments.append(cellprop)

    return experiments



def load_experiments_at_density(full_experiments, density, bin_size=200):
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



def load_correlation_data(DATASET_IDS=EXP_DATASETS_IDS):
    """ Load all correlations of all experiments """

    experiments = []
    for ID in DATASET_IDS:
        correlation = ExperimentalAutocorrelations(f"{EXP_PROC_DIR}/{ID}_cells.autocorr")
        experiments.append(correlation)

    return experiments



def load_correlations_at_density(full_correlations, density, bin_size=200):
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


