import sys
import pickle
from pathlib import Path
# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


from cells.bind     import VertexModel
from Correlations   import SimulationAutocorrelations


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
