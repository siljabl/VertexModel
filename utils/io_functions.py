import copy
import pickle
import numpy as np
from types import SimpleNamespace

from pathlib import Path
from cells.bind import VertexModel
from SegmentedCells import SegmentedCells
from paths_config import SIM_RAW_DIR, EXP_RAW_DIR, EXP_DATASETS_IDS


def load_simulation(file, init_time=100):
    """ Loads vm object and returns as list """
    
    list_vm = []
    init_vm = []

    time = 0
    with open(file, "rb") as dump:
        while True:
            try:
                vm = pickle.load(dump)
                assert type(vm) is VertexModel  # check pickled object is a vertex model
                
                if time < init_time:
                    init_vm += [vm]             # save first frames as init_vm
                else:
                    list_vm += [vm]             # append frame to list_vm

                time += 1

            except EOFError:
                break                           # stop when we have read the whole file

    return list_vm, init_vm



def load_simulation_ensemble(dirpath, Ngrid, init_time=100):

    dirpath = Path(dirpath)
    runs = []

    for path in sorted(dirpath.glob(f"N{Ngrid}_seed*.p")):
        list_vm, init_vm = load_simulation(path, init_time=init_time)
        runs.append(list_vm)

    return runs



def load_full_experiments(DATASET_IDS=EXP_DATASETS_IDS):

    experiments = []
    for ID in DATASET_IDS:
        cellprop = SegmentedCells(f"{EXP_RAW_DIR}/{ID}_cells.p")
        experiments.append(cellprop)

    return experiments



def load_experiments_at_density(full_experiments, density, bin_size=200):

    # Load full experiment of all datasets
    #full_experiments = load_full_experiments(DATASET_IDS)

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


# def load_experiments_by_density(densities, bin_size=200, DATASET_IDS=EXP_DATASETS_IDS):

#     # Load full experiment of all datasets
#     full_experiments = load_full_experiments(DATASET_IDS)

#     experiments_by_density = []

#     # Loop through densities and mask data
#     for density in densities:
#         experiments_at_density = []

#         for exp in full_experiments:

#             exp_by_density = copy.copy(exp)

#             min_density = density - bin_size / 2
#             max_density = density + bin_size / 2

#             mask = (exp.density > min_density) * (exp.density < max_density)

#             exp_by_density.filter_by_density(mask)
#             exp_by_density.drop_empty_cells()
        
#             experiments_at_density.append(exp_by_density)

#         experiments_by_density.append(experiments_at_density)

#     return experiments_by_density



