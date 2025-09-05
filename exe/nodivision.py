
import sys
import pickle
import argparse
import subprocess
import numpy as np
import scipy as sc

from pathlib  import Path
from tempfile import mkdtemp
from datetime import datetime

from cells.bind import VertexModel
from cells.plot import plot
from cells.init import movie_sh_fname

from utils.vm_functions       import *
from utils.exception_handlers import save_snapshot
from utils.config_functions   import load_config, save_config


# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run simulation without activity to relax the initial conditions")
parser.add_argument('--config', type=str, default='data/simulated/configs/config.json')
args = parser.parse_args()


# Load config file
config_path = args.config
config = load_config(config_path)

# Define paths for output
fname = f"{Path(__file__).stem}_{datetime.today().strftime('%Y%m%d_%H%M')}"

path_to_config = f"data/simulated/configs/{fname}.json"
path_to_output = f"data/simulated/raw/{fname}.p"
path_to_movies = f"data/simulated/videos/{fname}.p"

print("Simulation name: ", fname)

_frames_dir = mkdtemp()
print("Save frames to temp directory \"%s\"." % _frames_dir, file=sys.stderr)



# PARAMETERS

# Lattice
VMseed = config['simulation']['VMseed']                 # random number generator seed for vertex model object
Ngrid  = config['simulation']['Nvertices']              # number of vertices in each dimension. Ncell = Ngrid**2 / 3
r0     = config['simulation']['rgrid']                  # length scale of triangular lattice


# Cell size
rho   = config['physics']['rho']                        # defines compression/stretching of cells
rhex  = rho * r0                                        # reference side lenght of regular cell (hexagon)
A0    = hexagon_area(r0)                                # initial cell area
V0    = hexagon_volume(rhex)                            # cell volume
stdV0 = config['experimental']['stdV0'] * V0            # standard deviation of cell volume distribution
Vmin  = config['experimental']['Vmin']  * V0            # lower limit on volume
Vmax  = config['experimental']['Vmax']  * V0            # upper limit on volume
Vseed = config['simulation']['Vseed']                   # random number generator seed for volume distribution

# Forces
Lambda = config['physics']['Lambda']                    # surface tension
tauV   = config['physics']['tauV']                      # inverse increase rate in V0 unit
v0     = config['physics']['v0']                        # self-propulsion velocity
taup   = config['physics']['taup']                      # self-propulsion persistence time

# Integration
dt      = config['simulation']['dt']                    # integration time step
delta   = config['simulation']['delta']                 # length below which T1s are triggered
epsilon = config['simulation']['epsilon']               # edges have length delta+epsilon after T1s
period  = config['simulation']['period']                # saving frequence
Nsteps  = config['simulation']['Nsteps']                # number of steps/frames in simulation


# Save simulation-specific config file
save_config(path_to_config, config)



# INITIALISATION

# Vertex model object
vm = VertexModel(VMseed)                                        # initialise vertex model object
vm.initRegularTriangularLattice(size=Ngrid, hexagonArea=A0)     # initialise periodic system


# Add forces
np.random.seed(Vseed)
vm.addActiveBrownianForce("abp", v0, taup)                      # centre active Brownian force
vm.addSurfaceForce("surface", Lambda, V0, tauV)                 # surface tension force
vm.vertexForces["surface"].volume = dict(map(                   # set cell volume
    lambda i: (i, sc.stats.truncnorm((Vmin-V0)/stdV0, (Vmax-V0)/stdV0, loc=V0, scale=stdV0).rvs()),
    vm.vertexForces["surface"].volume))



# SIMULATION

# outputs
with open(path_to_output, "wb") as dump: pass           # output file is created
fig, ax = plot(vm, fig=None, ax=None)                   # initialise plot with first frame


# simulation
frame = 0
for step in range(0, Nsteps):
    # output is appended to file
    with open(path_to_output, "ab") as dump: pickle.dump(vm, dump)

    # plot snapshot
    save_snapshot(vm, fig, ax, _frames_dir, frame)
    frame += 1

    # integrate
    vm.nintegrate(period, dt, delta, epsilon)


# make movie
subprocess.call([movie_sh_fname,
                 "-d", _frames_dir,
                 "-o", path_to_movies,
                 "-p", sys.executable,
                 "-y"])



