import numpy as np
from operator import itemgetter
from cells.bind import BaseIntegrator
from utils.vm_calibration import cell_volume, division_probability, death_probability, cell_density
from utils.vm_observables import centre_indices


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

    dt = config["simulation"]["dt"]        # integration time step
    g  = config["calibration"]["g"]        # timescale of cell growth

    # Get cell volumes
    volumes = vm.vertexForces["surface"].volume.copy()

    for i in vm.getVertexIndicesByType("centre"):
        # linear growth
        volumes[i] += g*dt

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
    Performs cell division on vm object by splitting the dividing cell along its longest axis.
    After division the daughter cells will round up, thus mimicking the delayed behaviour in cells during mitosis

    Division probability for cell i:
        p_div = 
    """

    # Get cell volumes and cell heights
    volumes = vm.vertexForces["surface"].volume.copy()
    heights = vm.vertexForces["surface"].height.copy()

    counter = 0
    for i in vm.getVertexIndicesByType("centre"):

        # Division probability
        p_div = division_probability(config, volumes[i])
        if np.random.rand() < p_div:

            # Split cell i, get new vertice index j
            j = vm.splitCellAtMiddle(i, avoidThreeEdgeCells=True,
                                        avoidThinCells=True,
                                        edgeLim=config["simulation"]["edgeLim"])

            # Update volumes of daughter cells
            # Equal height
            # volumes[i] = heights[i]*vm.getVertexToNeighboursArea(i)
            # volumes[j] = heights[i]*vm.getVertexToNeighboursArea(j)

            # Equal volume
            volumes[i] = 0.5 * volumes[i]
            volumes[j] = volumes[i]

            counter += 1


    # Update cell volumes
    vm.vertexForces["surface"].volume = volumes


def cell_death(vm, config):
    """ 
    Performs apoptosis on vm object by merging the dying cell with the neighbour it shares the longest junction with. 

    Probability of dying for cell i:
        p_death = 
    """

    Ncells  = len(vm.getVertexIndicesByType("centre"))
    Ntarget = config["simulation"]["Nvertices"]**2 / 3
    p_dead = Ncells / Ntarget - 1

    for i in vm.getVertexIndicesByType("centre"):
        # Get area of cell i
        area = vm.getVertexToNeighboursArea(i)

        # Death probability
        # p_dead = death_probability(config, area)

        if np.random.rand() < p_dead:

            # # Skip if to few neighbours
            # N_neighbours = len(vm.getNeighbourVertices(i)[1])
            # if N_neighbours <= 4:
            #     continue

            j,  n_indices  = vm.mergeCellAtMax(i)
            # j,  n_indices  = vm.mergeCellAtMin(i)


def pulsating_cells(vm, config):
    # Placeholder: Sinusoidal A0, all with same amplitude and frequency, but phase from distribution
    # Paper of Li
    return 0
