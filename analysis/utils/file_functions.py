import pickle

from cells.bind import VertexModel


def load_vm_output(file):
    """ Loads vm object and returns as list """
    
    list_vm = []
    with open(file, "rb") as dump:
        while True:
            try:
                vm = pickle.load(dump)
                assert type(vm) is VertexModel  # check pickled object is a vertex model
                if vm.time == 0: 
                    init_vm = vm
                    continue       # do not use first frame
                list_vm += [vm]                 # append frame to list_vm
            except EOFError:
                break                           # stop when we have read the whole file

    return list_vm, init_vm
