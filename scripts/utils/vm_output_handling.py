import pickle
import numpy as np

from operator import itemgetter
from cells.bind import VertexModel, getPolygonsCell


def load(file, init_time=100, df=1):
    """ Loads vm object and returns as list """
    
    list_vm = []
    init_vm = []
    with open(file, "rb") as dump:
        while True:
            try:
                vm = pickle.load(dump)
                assert type(vm) is VertexModel  # check pickled object is a vertex model
                
                vm.nintegrate(1,0)              # integrate so output corresponds to correct frame/timestep 
                if vm.time < init_time * df: 
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



def get_neighbour_matrix(list_vm):
    """ Get cell volumes """

    # indices of cell centres (from first frame)
    cells = list_vm[0].getVertexIndicesByType("centre")

    neighbours_matrix = np.zeros([len(list_vm), max(cells)+1, max(cells)+1])

    for frame in range(len(list_vm)):
        for cell in cells:
            neighbours = list_vm[frame].getNeighbouringCellIndices(cell)
            neighbours_matrix[frame, cell, neighbours] = 1


    return neighbours


def get_cell_aspect_ratios(list_vm):

    cells = list_vm[0].getVertexIndicesByType("centre")

    aspect_ratio = []

    for vm in list_vm:  
        polygons = getPolygonsCell(vm)
        centers  = itemgetter(*cells)(list_vm[0].getPositions(wrapped=True))

        centered_polygons = [np.array(polygon) - np.array(center) for polygon, center in zip(polygons, centers)]


        cov = [np.cov(polygon, rowvar=False, bias=False) for polygon in polygons]

        # eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)  # returns sorted (ascending) eigenvalues for symmetric matrices

        # eigenvalues are in ascending order (smallest first), so:
        minor_idx = 0
        major_idx = 1

        # directions (unit vectors)
        minor_dir = eigvecs[:, minor_idx]
        major_dir = eigvecs[:, major_idx]

        proj_major = [np.array(polygon).dot(major) for polygon, major in zip(polygons, major_dir)]
        proj_minor = [np.array(polygon).dot(minor) for polygon, minor in zip(polygons, minor_dir)]

        major_length = np.array([major.max() - major.min() for major in proj_major])
        minor_length = np.array([minor.max() - minor.min() for minor in proj_minor])

        aspect_ratio.append(major_length / minor_length)


    return np.array(aspect_ratio)

