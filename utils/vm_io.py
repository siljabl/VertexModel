import sys
import copy
import pickle
from pathlib import Path
# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import utils.config_io as cfg
from cells.bind import VertexModel
from utils.Correlations import SimulationAutocorrelations



def load_run(file, init_time):
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



def load_ensemble(dirpath, rho=None, n_vertices=None):
    """ Loads all vm objects in ensemble and returns as list """

    assert rho or n_vertices

    if rho:
        pattern = f"rho{rho}"
    else:
        pattern = f"N{n_vertices}"


    dirpath = Path(dirpath)
    runs = []

    for path in sorted(dirpath.glob(f"{pattern}_seed*.p")):
        cfg_path = (Path("configs") / path.parent.parent.stem / path.parent.stem / path.stem).with_suffix(".json")
        config = cfg.load(cfg_path)

        list_vm, init_vm = load_run(path, init_time=config["integration"]["n_transient"])
        runs.append(list_vm)

    return runs



def load_ensemble_correlations(dirpath, rho=None, n_vertices=None):
    """ Loads all correlation objects in ensemble and returns as list """

    assert rho or n_vertices

    if rho:
        pattern = f"rho{rho}"
    else:
        pattern = f"N{n_vertices}"

    dirpath = Path(dirpath)
    runs = []

    for path in sorted(dirpath.glob(f"{pattern}_seed*.p")):
        corr = SimulationAutocorrelations(in_path=path)

        runs.append(corr)

    return runs



