import pickle
import numpy as np

from operator import itemgetter
from cells.bind import VertexModel, getPolygonsCell

def centre_indices(list_vm):
    return list_vm[0].getVertexIndicesByType("centre")


def cell_positions(list_vm):
    """ Get cell positions """

    # indices of cell centres (from first frame)
    cells = centre_indices(list_vm)

    # unwrap positions of centres
    positions = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.getPositions(wrapped=False)), 
        list_vm)))

    return positions


def cell_heights(list_vm):
    """ Get cell heights """

    # indices of cell centres (from first frame)
    cells = centre_indices(list_vm)

    # unwrap cell heights
    heights = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.vertexForces["surface"].height.copy()),
        list_vm)))

    return heights


def cell_volumes(list_vm):
    """ Get cell volumes """

    # indices of cell centres (from first frame)
    cells = centre_indices(list_vm)

    # unwrap cell volumes
    volumes = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.vertexForces["surface"].volume.copy()), 
        list_vm)))

    return volumes


def cell_areas(list_vm):
    h = cell_heights(list_vm)
    V = cell_volumes(list_vm)

    return V / h


def cell_perimeters(list_vm):
    """ Get cell perimeters"""

    perimeter = []
    for vm in list_vm:
        perimeter.append(np.array(list(map(
                         lambda i: vm.getVertexToNeighboursPerimeter(i),
                         vm.getVertexIndicesByType("centre"))))) 

    return np.ma.array(perimeter)



def cell_displacements(list_vm):

    # dt = list_vm[1].time - list_vm[0].time

    positions     = cell_positions(list_vm)
    vector_displacements = np.ma.diff(positions, axis=0)
    displacements = np.ma.sqrt(vector_displacements[:,:,0]**2 + vector_displacements[:,:,1]**2)

    return displacements





def cell_velocities(list_vm):
    """ Get cell volumes """

    # indices of cell centres (from first frame)
    cells = centre_indices(list_vm)

    # unwrap cell velocities at cell centers
    velocities = np.ma.array(list(map(
        #lambda vm: itemgetter(*cells)(vm.velocities.copy()), #.getCentreVelocities()
        lambda vm: itemgetter(*cells)(vm.getCentreVelocities().copy()),
        list_vm)))

    return velocities


def cell_speeds(list_vm):
    """ Get cell volumes """

    # unwrap cell velocities at cell centers
    velocities = cell_velocities(list_vm)
    speeds     = np.ma.sqrt(velocities[:,:,0]**2 + velocities[:,:,1]**2)    

    return speeds



def neighbour_matrix(list_vm):
    """ Get cell volumes """

    # indices of cell centres (from first frame)
    cells = centre_indices(list_vm)

    neighbours_matrix = np.zeros([len(list_vm), max(cells)+1, max(cells)+1])

    for frame in range(len(list_vm)):
        for cell in cells:
            neighbours = list_vm[frame].getNeighbouringCellIndices(cell)
            neighbours_matrix[frame, cell, neighbours] = 1

    return neighbours


def cell_planar_aspect_ratios(list_vm):

    # cells = centre_indices(list_vm)

    aspect_ratios = []

    for vm in list_vm:  
        polygons = getPolygonsCell(vm)
        # centers  = itemgetter(*cells)(list_vm[0].getPositions(wrapped=True))

        #centered_polygons = [np.array(polygon) - np.array(center) for polygon, center in zip(polygons, centers)]

        amajor = []
        aminor = []

        for polygon in polygons:
            cov = np.cov(polygon, rowvar=False, bias=False)

            # eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)

            # eigenvalues are in ascending order (smallest first), so:
            # directions (unit vectors)
            minor_dir = eigvecs[:, 0]
            major_dir = eigvecs[:, 1]

            proj_major = polygon @ major_dir #[np.array(polygon).dot(major) for polygon, major in zip(polygons, major_dir)]
            proj_minor = polygon @ minor_dir #[np.array(polygon).dot(minor) for polygon, minor in zip(polygons, minor_dir)]

            major_length = proj_major.max() - proj_major.min()
            minor_length = proj_minor.max() - proj_minor.min()

            amajor.append(major_length)
            aminor.append(minor_length)

        aspect_ratios.append(np.array(amajor) / np.array(aminor))


    return np.array(aspect_ratios)



def cell_vertical_aspect_ratios(list_vm):

    heights = cell_heights(list_vm)
    areas   = cell_areas(list_vm)


    return np.array(heights / np.ma.sqrt(areas))



def ensemble_observable(vms, func):
    """
    vms  : list of runs, each run is list_vm (frames)
    func : function(list_vm) -> array (T, N)

    Returns
    -------
    obs : np.ma.array
        Shape (T, R, N): frame, run, cell.
    """

    obs = []

    for vm in vms:
        observable = func(vm)
        obs.append(observable)

    obs = np.ma.array(obs)

    return np.swapaxes(obs, 0, 1)