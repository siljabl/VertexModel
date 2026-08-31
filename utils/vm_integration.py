import numpy as np
from cells.bind import BaseIntegrator
from utils.vm_calibration import cell_division_volume, cell_volume

def one_timestep(vm, config):

    dt      = config['simulation']['dt']
    delta   = config['simulation']['delta'] 
    epsilon = config['simulation']['epsilon']

    vm.integrate(dt, delta, epsilon)

    return 0


    
def cell_growth(vm, config):

    dt   = config["simulation"]["dt"]
    tauV = config["physics"]["tauV"]

    volumes = vm.vertexForces["surface"].volume.copy()

    for i in vm.getVertexIndicesByType("centre"):
        # linear growth
        volumes[i] += dt / tauV

    vm.vertexForces["surface"].volume = volumes

    return 0


def volume_relaxation(vm, config):

    dt   = config["simulation"]["dt"]
    tauV = config["physics"]["tauV"]

    V0      = cell_volume(config)
    volumes = vm.vertexForces["surface"].volume.copy()

    for i in vm.getVertexIndicesByType("centre"):
        # exponential relaxation
        volumes[i] += (V0 - volumes[i]) * dt / tauV

    vm.vertexForces["surface"].volume = volumes

    return 0



def cell_division(vm, config):
    """ Performs cell division on vm object """
    
    volumes = vm.vertexForces["surface"].volume.copy()
    heights = vm.vertexForces["surface"].height.copy()

    Vth = cell_division_volume(config)

    for i in vm.getVertexIndicesByType("centre"):

        if np.random.rand() < (volumes[i] - Vth)/Vth:

            j = vm.splitCellAtMax(i)
            volumes[i] = heights[i]*vm.getVertexToNeighboursArea(i)
            volumes[j] = heights[i]*vm.getVertexToNeighboursArea(j)

    vm.vertexForces["surface"].volume = volumes

    return vm


def pulsating_cells(v, config):
    return 0
