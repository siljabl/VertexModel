import numpy as np
from cells.bind import BaseIntegrator
from utils.vm_calibration import cell_division_volume, cell_volume


def one_timestep(vm, config):
    """
    Perform one integration step of the VertexModel.
    """

    dt      = config['simulation']['dt']        # integration time step
    delta   = config['simulation']['delta']     # length below which T1s are triggered
    epsilon = config['simulation']['epsilon']   # edges have length delta+epsilon after T1s

    vm.integrate(dt, delta, epsilon)


    
def cell_growth(vm, config):
    """
    Linear growth of cell volumes: dV/dt = 1 / tauV.
    """

    dt   = config["simulation"]["dt"]        # integration time step
    tauV = config["physics"]["tauV"]         # timescale of cell growth

    # Get cell volumes
    volumes = vm.vertexForces["surface"].volume.copy()

    for i in vm.getVertexIndicesByType("centre"):
        # linear growth
        volumes[i] += dt / tauV

    # Update cell volumes
    vm.vertexForces["surface"].volume = volumes



def volume_relaxation(vm, config):
    """
    Exponential relaxation of volumes towards V0:
        dV/dt = (V0 - V) / tauV
    """

    dt   = config["simulation"]["dt"]   # integration time step
    tauV = config["physics"]["tauV"]    # relaxation timescale

    # Experimental mean cell volume (from calibration / density)
    V0      = cell_volume(config)

    # Get cell volumes
    volumes = vm.vertexForces["surface"].volume.copy()

    for i in vm.getVertexIndicesByType("centre"):
        # exponential relaxation
        volumes[i] += (V0 - volumes[i]) * dt / tauV

    # Update cell volumes
    vm.vertexForces["surface"].volume = volumes



def cell_division(vm, config):
    """ 
    Performs cell division on vm object 
    
    Division probability for cell i:
        p_div = 

    """

    # Get cell volumes and cell heights
    volumes = vm.vertexForces["surface"].volume.copy()
    heights = vm.vertexForces["surface"].height.copy()

    # 
    Vth = cell_division_volume(config)

    for i in vm.getVertexIndicesByType("centre"):

        # Division probability
        p_div = (volumes[i] - Vth)/Vth
        if np.random.rand() < p_div:

            # Split cell i, get new vertice index j
            j = vm.splitCellAtMax(i)

            # Update volumes of daughter cells 
            volumes[i] = heights[i]*vm.getVertexToNeighboursArea(i)
            volumes[j] = heights[i]*vm.getVertexToNeighboursArea(j)

    # Update cell volumes
    vm.vertexForces["surface"].volume = volumes

    # return vm


def pulsating_cells(vm, config):
    # Placeholder: Sinusoidal A0, all with same amplitude and frequency, but phase from distribution
    # Paper of Li
    return 0
