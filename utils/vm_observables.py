import pickle
import numpy as np

from operator import itemgetter
from cells.bind import getPolygonsCell


# NB: Update when adding cell division!
def centre_indices(list_vm):
    """Return indices of cell centres (from first frame)."""
    return list_vm[0].getVertexIndicesByType("centre")



def cell_positions(list_vm):
    """Get cell positions at centres for each frame."""

    cells = centre_indices(list_vm)

    # unwrap positions of centres
    positions = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.getPositions(wrapped=False)), 
        list_vm)))

    return positions



def cell_heights(list_vm):
    """ Get cell heights """

    cells = centre_indices(list_vm)

    # unwrap cell heights
    heights = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.vertexForces["surface"].height.copy()),
        list_vm)))

    return heights



def cell_volumes(list_vm):
    """ Get cell volumes """

    cells = centre_indices(list_vm)

    # unwrap cell volumes
    volumes = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.vertexForces["surface"].volume.copy()), 
        list_vm)))

    return volumes



def cell_areas(list_vm):
    """ Get cell areas """
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



def cell_displacement_vectors(list_vm):
    """Get cell displacement vectors per frame (vx, vy)."""

    # Get simulation time step
    dt = list_vm[1].time - list_vm[0].time

    position_vectors     = cell_positions(list_vm)
    displacement_vectors = np.ma.diff(position_vectors, axis=0)

    return displacement_vectors / dt



def cell_velocities(list_vm):
    """ Get instantaneous cell velocities """

    # indices of cell centres (from first frame)
    cells = centre_indices(list_vm)

    # unwrap cell velocities at cell centers
    velocities = np.ma.array(list(map(
        lambda vm: itemgetter(*cells)(vm.getCentreVelocities().copy()),
        list_vm)))

    return velocities



def cell_vector_norm(vector):
    """ Take norm of vector in cell observable """
    return np.ma.sqrt(vector[:,:,0]**2 + cell_vector_norm[:,:,1]**2) 



def cell_displacements(list_vm):
    """Get scalar cell displacements per frame (distance per unit time)."""

    displacement_vectors = cell_displacement_vectors(list_vm)
    scalar_displacements = cell_vector_norm(displacement_vectors)

    return scalar_displacements



def cell_speeds(list_vm):
    """ Get instantenaous cell speed """

    # unwrap cell velocities at cell centers
    velocities = cell_velocities(list_vm)
    speeds     = cell_vector_norm(velocities) 

    return speeds



def neighbour_matrix(list_vm):
    """
    Get neighbour matrix over time.

    Returns
    -------
    neighbours_matrix : np.ndarray
        Shape (T, Ncells_max+1, Ncells_max+1), with entries 1 if cells
        are neighbours, 0 otherwise, for each frame.
    """

    cells = centre_indices(list_vm)

    neighbours_matrix = np.zeros([len(list_vm), max(cells) + 1, max(cells) + 1])

    for frame in range(len(list_vm)):
        for cell in cells:
            neighbours = list_vm[frame].getNeighbouringCellIndices(cell)
            neighbours_matrix[frame, cell, neighbours] = 1

    return neighbours_matrix



def cell_planar_aspect_ratios(list_vm):
    """
    Get planar cell aspect ratios (major/minor) for each frame,
    based on eigenvalues of the covariance of polygon vertices.
    """

    aspect_ratios = []

    for vm in list_vm:  
        polygons = getPolygonsCell(vm)

        amajor = []
        aminor = []

        for polygon in polygons:
            cov = np.cov(polygon, rowvar=False, bias=False)

            # Eigen-decomposition
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Eigenvalues are in ascending order (smallest first)
            minor_dir = eigvecs[:, 0]
            major_dir = eigvecs[:, 1]

            # Project polygon onto unit vectors
            proj_major = polygon @ major_dir
            proj_minor = polygon @ minor_dir

            # Compute axes lengths
            major_length = proj_major.max() - proj_major.min()
            minor_length = proj_minor.max() - proj_minor.min()

            # Collect axes of cell
            amajor.append(major_length)
            aminor.append(minor_length)

        # Collect aspect ratio of all cells in frame
        aspect_ratios.append(np.array(amajor) / np.array(aminor))

    return np.array(aspect_ratios)



def cell_vertical_aspect_ratios(list_vm):
    """ Get vertical aspect ratios"""

    heights = cell_heights(list_vm)
    areas   = cell_areas(list_vm)

    return np.array(heights / np.ma.sqrt(areas))