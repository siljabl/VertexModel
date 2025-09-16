import pickle
import numpy as np

from operator import itemgetter
from cells.bind import VertexModel


def load(file, init_time=100):
    """ Loads vm object and returns as list """
    
    list_vm = []
    init_vm = []
    with open(file, "rb") as dump:
        while True:
            try:
                vm = pickle.load(dump)
                assert type(vm) is VertexModel  # check pickled object is a vertex model
                
                vm.nintegrate(1,0)              # integrate so output corresponds to correct frame/timestep 
                if vm.time < init_time: 
                    init_vm += [vm]             # save first frames as init_vm
                else:
                    list_vm += [vm]             # append frame to list_vm
            except EOFError:
                break                           # stop when we have read the whole file

    return list_vm, init_vm



def get_cell_positions(list_vm):
    """ Get cell positions """

    # indices of cell centres (from first frame)
    cells = list_vm[0].getVertexIndicesByType("centre")

    # unwrap positions of centres
    positions = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.getPositions(wrapped=False)), 
        list_vm)))

    return positions


def get_cell_heights(list_vm):
    """ Get cell heights """

    # indices of cell centres (from first frame)
    cells = list_vm[0].getVertexIndicesByType("centre")

    # unwrap cell heights
    heights = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.vertexForces["surface"].height.copy()),
        list_vm)))

    return heights


def get_cell_volumes(list_vm):
    """ Get cell volumes """

    # indices of cell centres (from first frame)
    cells = list_vm[0].getVertexIndicesByType("centre")

    # unwrap cell volumes
    volumes = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.vertexForces["surface"].volume.copy()), 
        list_vm)))

    return volumes


def get_cell_velocities(list_vm):
    """ Get cell volumes """

    # indices of cell centres (from first frame)
    cells = list_vm[0].getVertexIndicesByType("centre")

    # unwrap cell velocities at cell centers
    velocities = np.ma.array(list(map(
        #lambda vm: itemgetter(*cells)(vm.velocities.copy()), #.getCentreVelocities()
        lambda vm: itemgetter(*cells)(vm.getCentreVelocities().copy()),
        list_vm)))

    return velocities