import numpy as np
from operator import itemgetter
from cells.bind import BaseIntegrator
from utils.vm_calibration import cell_volume, division_probability
from utils.vm_observables import centre_indices


def one_timestep(vm, config):
    """
    Perform one integration step of the VertexModel.
    """

    dt      = config['integration']['dt']
    delta   = config['simulation']['t1_threshold']
    epsilon = config['simulation']['t1_offset']

    vm.integrate(dt, delta, epsilon)


    
def cell_growth(vm, config):
    """
    Linear growth of cell volumes: dV/dt = g.
    """

    dt = config["integration"]["dt"]
    g  = config["growth"]["g"]

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
        dV/dt = g(V0 - V)
    """

    dt = config["integration"]["dt"]
    g  = config["growth"]["g"]

    # Experimental mean cell volume (from calibration / density)
    V0      = cell_volume(config)

    # Get cell volumes
    volumes = vm.vertexForces["surface"].volume.copy()

    for i in vm.getVertexIndicesByType("centre"):
        # exponential relaxation
        volumes[i] += (V0 - volumes[i]) * g * dt

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

    for i in vm.getVertexIndicesByType("centre"):

        # Division probability
        p_div = division_probability(config, volumes[i])
        if np.random.rand() < p_div:

            # Split cell i, get new vertice index j
            j = vm.splitCellAtMiddle(i, avoidThreeEdgeCells=True,
                                        avoidThinCells=True,
                                        edgeLim=config["simulation"]["edge_limit"])

            # Update volumes of daughter cells to have half of mother volume
            volumes[i] = 0.5 * volumes[i]
            volumes[j] = volumes[i]

    # Update cell volumes
    vm.vertexForces["surface"].volume = volumes



def cell_death(vm, config):
    """ 
    Performs apoptosis on vm object by merging the dying cell with the neighbour it shares the longest junction with. 

    Probability of dying for cell i:
        p_death = 
    """

    Ncells  = len(vm.getVertexIndicesByType("centre"))
    Ntarget = 1e6 / config["cells"]["target_density"]
    p_dead = Ncells / Ntarget - 1

    for i in vm.getVertexIndicesByType("centre"):
        # Get area of cell i
        area = vm.getVertexToNeighboursArea(i)

        # Death probabilityimulation
        # p_dead = death_probability(config, area)

        if np.random.rand() < p_dead:

            # Merge cells
            j,  n_indices  = vm.mergeCellAtMax(i)


def pulsating_cells(vm, config):
    # Placeholder: Sinusoidal A0, all with same amplitude and frequency, but phase from distribution
    # Paper of Li
    return 0
