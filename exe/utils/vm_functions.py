from cells.bind import VertexModel
import numpy as np


def cell_divisions(vm, Vth):
    volumes = vm.vertexForces["surface"].volume.copy()
    heights = vm.vertexForces["surface"].height.copy()

    for i in vm.getVertexIndicesByType("centre"):

        if np.random.rand() < (volumes[i] - Vth)/Vth:

            j = vm.SplitCellAtMax(i)
            volumes[i] = heights[i]*vm.getVertexToNeighboursArea(i)
            volumes[j] = heights[i]*vm.getVertexToNeighboursArea(j)

        vm.vertexForces["surface"].volume = volumes

    return vm

