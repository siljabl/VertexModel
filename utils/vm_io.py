import pickle

from pathlib import Path
from cells.bind import VertexModel


def load(file, init_time=100):
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



def load_ensemble(dirpath, Ngrid, init_time=100):

    dirpath = Path(dirpath)
    runs = []

    for path in sorted(dirpath.glob(f"N{Ngrid}_seed*.p")):
        list_vm, init_vm = load(path, init_time=init_time)
        runs.append(list_vm)

    return runs
