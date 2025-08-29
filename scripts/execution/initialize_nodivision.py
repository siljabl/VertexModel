import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cells.bind import VertexModel
from cells.plot import plot
from cells.init import movie_sh_fname

from utils.exception_handlers import save_snapshot

import pickle, subprocess, traceback
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from tempfile import mkdtemp
from datetime import datetime


# set ouput paths
fname = f"init_nodivision_{datetime.today().strftime('%Y%m%d')}"
path_to_output = f"data/simulated/raw/{fname}.p"
path_to_movies = f"data/simulated/videos/{fname}.p"

print("Simulation name: ", fname)


# PARAMETERS
seed = 0                                # random number generator seed
Nsteps = 30
N = 24                                  # number of vertices in each dimension

#v0 = 0                                  # self-propulsion velocity
#taup = 1                                # self-propulsion persistence time

Lambda = 1                              # surface tension
V0 = 1                                  # reference volume of cells
Vth = 1.5*V0                            # threshold volume
A0 = (np.sqrt(3)*(V0**2)/2)**(1./3.)    # reference area of cells
stdV0 = 0.75                            # standard deviation of volume of cells
tauV = 0                              # inverse increase rate in V0 unit



# INITIALISATION

# vertex model object
vm = VertexModel(seed)                                  # initialise vertex model object
vm.initRegularTriangularLattice(size=N, hexagonArea=A0) # initialise periodic system

# forces
#vm.addActiveBrownianForce("abp", v0, taup)      # centre active Brownian force
vm.addSurfaceForce("surface", Lambda, V0, tauV) # surface tension force
vm.vertexForces["surface"].volume = dict(map(   # set cell volume
    lambda i: (i, np.random.uniform(low=V0 - stdV0, high=V0 + stdV0)),
    vm.vertexForces["surface"].volume))



# SIMULATION

# parameters
dt = 1e-2           # integration time step
delta = 0.02        # length below which T1s are triggered
epsilon = 0.002     # edges have length delta+epsilon after T1s
period  = 2         # saving frequence

# frames directory
_frames_dir = mkdtemp()
print("Save frames to temp directory \"%s\"." % _frames_dir, file=sys.stderr)
frame = 0

# output
with open(path_to_output, "wb") as dump: pass           # output file is created

# simulation
fig, ax = plot(vm, fig=None, ax=None)                   # initialise plot with first frame

for step in range(0, Nsteps):
    # output is appended to file
    with open(path_to_output, "ab") as dump: pickle.dump(vm, dump)

    # plot
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



