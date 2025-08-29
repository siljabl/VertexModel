import os, sys, pickle, subprocess, traceback, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cells.bind import VertexModel
from cells.plot import plot
from cells.init import movie_sh_fname

from utils.exception_handlers import save_snapshot
from utils.config_functions   import load_config, save_config

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from tempfile import mkdtemp
from datetime import datetime


# command-line argument parsing
parser = argparse.ArgumentParser(description="Run simulation without activity to relax the initial conditions")
parser.add_argument('--config', type=str, default='data/simulated/configs/config.json')
args = parser.parse_args()


# load existing configuration
config_path = args.config
config = load_config(config_path)


# set ouput paths
fname = f"init_nodivision_{datetime.today().strftime('%Y%m%d')}"
path_to_output = f"data/simulated/raw/{fname}.p"
path_to_movies = f"data/simulated/videos/{fname}.p"
print("Simulation name: ", fname)

_frames_dir = mkdtemp()
print("Save frames to temp directory \"%s\"." % _frames_dir, file=sys.stderr)



# PARAMETERS

seed = config['simulation']['seed']                     # random number generator seed
N    = config['simulation']['Nvertices']                # number of vertices in each dimension

Lambda = config['physics']['Lambda']                    # surface tension
V0     = config['physics']['V0']                        # reference volume of cells
Vth    = config['physics']['Vth/V0'] * V0               # threshold volume
stdV0  = 0.5                                            # standard deviation of volume of cells
tauV   = config['physics']['tauV']                      # inverse increase rate in V0 unit
A0     = (np.sqrt(3)*(V0**2)/2)**(1./3.)                # reference area of cells

dt      = config['simulation']['dt']                    # integration time step
delta   = config['simulation']['delta']                 # length below which T1s are triggered
epsilon = config['simulation']['epsilon']               # edges have length delta+epsilon after T1s
period  = config['simulation']['period']                # saving frequence
Nsteps  = config['simulation']['Nsteps']                # don't understand exactly what this is.



# INITIALISATION

# vertex model object
vm = VertexModel(seed)                                  # initialise vertex model object
vm.initRegularTriangularLattice(size=N, hexagonArea=A0) # initialise periodic system


# add forces
#vm.addActiveBrownianForce("abp", v0, taup)             # centre active Brownian force
vm.addSurfaceForce("surface", Lambda, V0, tauV)         # surface tension force
vm.vertexForces["surface"].volume = dict(map(           # set cell volume
    lambda i: (i, np.random.normal(loc=V0, scale=stdV0)),
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



