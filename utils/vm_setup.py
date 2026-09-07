import numpy as np
import scipy as sc
from utils.vm_geometry import hexagon_area
from utils.vm_calibration import cell_volume



def initialise_vm_lattice(vm, config):
    """
    Initialise VertexModel lattice as a regular triangular lattice.
    """
    Ngrid = config['simulation']['Nvertices']           # Number of vertices in each dimension. Ncell = Ngrid**2 / 3
    Lgrid = config['simulation']['Lgrid']               # Length of lattice in µm

    rgrid = Lgrid / Ngrid                               # Lattice spacing of triangular grid
    A0    = hexagon_area(rgrid)                         # Target average cell area

    vm.initRegularTriangularLattice(size=Ngrid, hexagonArea=A0)

    return vm



def set_cell_volumes(vm, config):
    """
    Initialise cell volumes drawn from a lognormal distribution 
    """

    # Lognormal parameters from calibration
    s     = config['calibration']['s']            # shape parameter (std. dev of log)
    scale = config['calibration']['scale']        # scale parameter (exp(mean of log))

    # Experimental mean cell volume (from calibration / density)
    V0     = cell_volume(config)

    # Mean of lognormal with parameters (s, scale) is: E[X] = scale * exp(s**2 / 2)
    Vmean  = scale * np.exp(s**2 / 2)

    # Choose Vscale so that mean(Vscale * X) = V0
    Vscale = V0 / Vmean
    
    # Set volume of each cell by drawing from the rescaled lognormal
    vm.vertexForces["surface"].volume = dict(map(
        lambda i: (i, Vscale * sc.stats.lognorm(s, scale=scale).rvs()),
        vm.vertexForces["surface"].volume))

    return vm



def initialise_vm_forces(vm, config):
    """
    Initialise forces in the VertexModel
    """
    gamma  = config['physics']['gamma']
    Lambda = config['physics']['lambda']
    tauV   = config['physics']['tauV']
    v0     = config['physics']['v0']
    taup   = config['physics']['taup']
    eta    = config['physics']['eta']

    # Reference volume (experimental mean) from calibration
    V0 = cell_volume(config)

    # Centre active Brownian force
    vm.addActiveBrownianForce("abp", v0, taup)

    # Surface tension force
    vm.addSurfaceForce("surface", gamma, Lambda, V0, tauV)

    # Pair dissipation
    vm.setPairFrictionIntegrator(eta)

    return vm


