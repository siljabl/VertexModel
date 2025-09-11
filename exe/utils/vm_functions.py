from cells.bind import VertexModel
import numpy as np


def cell_divisions(vm, Vth):
    """ Performs cell division on vm object """
    
    volumes = vm.vertexForces["surface"].volume.copy()
    heights = vm.vertexForces["surface"].height.copy()

    for i in vm.getVertexIndicesByType("centre"):

        if np.random.rand() < (volumes[i] - Vth)/Vth:

            j = vm.SplitCellAtMax(i)
            volumes[i] = heights[i]*vm.getVertexToNeighboursArea(i)
            volumes[j] = heights[i]*vm.getVertexToNeighboursArea(j)

        vm.vertexForces["surface"].volume = volumes

    return vm


def hexagon_volume(rhex):
    """ Computes volume of regular hexagon with side length rhex  """

    return (3**2 / 2) * rhex**3


def hexagon_area(rhex):
    """ Computes area of regular hexagon with side length rhex """

    return (3**(3/2) / 2) * (rhex)**2


def cell_density(Ngrid, Lgrid):
    """ Computes the global cell density """
    Ncell = Ngrid**2 / 3
    Agrid = (np.sqrt(3) / 2) * Lgrid**2

    return Ncell / Agrid
