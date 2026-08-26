# utils/vm_setup.py

import numpy as np
import scipy as sc
from cells.bind import VertexModel
from utils.vm_functions import hexagon_area, cell_volume



def initalise_vm_lattice(vm, config):
    """
    Initializes VertexModel lattice and returns (vm, config_with_rho_A0_V0).
    """
    Ngrid = config['simulation']['Nvertices']
    Lgrid = config['simulation']['Lgrid']

    rgrid = Lgrid / Ngrid
    A0    = hexagon_area(rgrid)

    vm.initRegularTriangularLattice(size=Ngrid, hexagonArea=A0)

    return vm



def set_cell_volumes(vm, config):

    Ngrid = config['simulation']['Nvertices']
    Lgrid = config['simulation']['Lgrid']
    s     = config['experimental']['s']                     # parameter of scipy.stats.lognorm
    scale = config['experimental']['scale']                 # parameter of scipy.stats.lognorm

    V0    = cell_volume(Ngrid, Lgrid)                       # cell volume

    vm.vertexForces["surface"].volume = dict(map(                   # set cell volume
        lambda i: (i, V0 / np.exp(np.log(scale) + s**2/2) * sc.stats.lognorm(s, scale=scale).rvs()),
        vm.vertexForces["surface"].volume))

    return vm



def initialise_vm_forces(vm, config):

    gamma  = config['physics']['gamma']
    Lambda = config['physics']['lambda']
    tauV   = config['physics']['tauV']
    v0     = config['physics']['v0']
    taup   = config['physics']['taup']
    eta    = config['physics']['eta']

    Ngrid = config['simulation']['Nvertices']
    Lgrid = config['simulation']['Lgrid']

    V0    = cell_volume(Ngrid, Lgrid)

    vm.addActiveBrownianForce("abp", v0, taup)                     # centre active Brownian force
    vm.addSurfaceForce("surface", gamma, Lambda, V0, tauV)         # surface tension force
    vm.setPairFrictionIntegrator(eta)                              # add pair dissipation

    return vm


