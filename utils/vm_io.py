import sys
import copy
import pickle
from pathlib import Path
# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from cells.bind import VertexModel
from utils.vm_calibration import cell_density_from_Ngrid, cell_density_from_Ncell
from utils.vm_observables import number_of_cells
from utils.Correlations import SimulationAutocorrelations



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



def load_simulation_at_density(dirpath, density, Lgrid=600, bin_size=200, init_time=100):
    """ Load all experiments in given density range """

    dirpath = Path(dirpath)
    runs_at_density = []

    min_density = density - bin_size / 2
    max_density = density + bin_size / 2

    for path in sorted(dirpath.glob(f"N*_seed*.p")):
        list_vm, init_vm = load_simulation(path, init_time=init_time)

        # Compute cell density during simulation
        N_cells   = number_of_cells(list_vm)
        densities = cell_density_from_Ncell(N_cells, Lgrid=Lgrid)

        mask = (densities > min_density) * (densities < max_density)
        run = []
        for vm, m in zip(list_vm, mask):
            if m:
                run.append(vm)

        runs_at_density.append(run)

    return runs_at_density



def load_simulation_ensemble(dirpath, Ngrid, init_time=100):
    """ Loads all vm objects in ensemble and returns as list """

    dirpath = Path(dirpath)
    runs = []

    for path in sorted(dirpath.glob(f"N{Ngrid}_seed*.p")):
        list_vm, init_vm = load_simulation(path, init_time=init_time)
        runs.append(list_vm)

    return runs



def load_correlation_ensemble(dirpath, Ngrid):
    """ Loads all correlation objects in ensemble and returns as list """


    dirpath = Path(dirpath)
    runs = []

    for path in sorted(dirpath.glob(f"N{Ngrid}_seed*.p")):
        corr = SimulationAutocorrelations(in_path=path)

        runs.append(corr)

    return runs
