from cells.bind import VertexModel
from cells.read import _progressbar as progressbar
from cells.plot import plot, WindowClosedException

import numpy as np
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt

# PARAMETERS

seed = 0    # random number generator seed

N = 12      # number of vertices in each dimension

v0 = 1.5    # self-propulsion velocity
taup = 1    # self-propulsion persistence time

Lambda = 1  # surface tension
A0 = 1      # mean area of cells
A6s = 1     # target area of cells in surface tension potential

# INITIALISATION

# vertex model object
vm = VertexModel(seed)                                  # initialise vertex model object
vm.initRegularTriangularLattice(size=N, hexagonArea=A0) # initialise periodic system

# forces
vm.addActiveBrownianForce("abp", v0, taup)                              # centre active Brownian force
vm.addSurfaceForce("surface", Lambda, np.sqrt(2*(A6s**3)/np.sqrt(3)))   # surface tension force

# SIMULATION

# parameters
dt = 1e-2           # integration time step
delta = 0.02        # length below which T1s are triggered
epsilon = 0.002     # edges have length delta+epsilon after T1s
init = 10000        # number of initialisation time steps
niter = 10000       # total number of production time steps
period = 100        # saving frequence

# simulation
assert niter%period == 0                            # number of steps should be multiple of frequence
with open("vm_output.p", "wb") as dump:
    progressbar(0)
    pickle.dump(vm, dump)                           # save first frame
    vm.nintegrate(init, dt, delta, epsilon)         # initialisation
    for iteration in range(niter//period):
        progressbar(iteration/(niter//period))
        pickle.dump(vm, dump)                       # save at start of frame
        vm.nintegrate(period, dt, delta, epsilon)   # run
    progressbar(1)
    pickle.dump(vm, dump)                           # save last frame

# ANALYSIS

# load frames
list_vm = []
with open("vm_output.p", "rb") as dump:
    while True:
        try:
            vm = pickle.load(dump)
            assert type(vm) is VertexModel  # check pickled object is a vertex model
            if vm.time == 0: continue       # do not use first frame
            list_vm += [vm]                 # append frame to list_vm
        except EOFError:
            break                           # stop when we have read the whole file

# visualise
progressbar(0)
fig, ax = plot(list_vm[0], fig=None, ax=None)   # initialise plot with first frame
plt.ion()                                       # enable interactive mode
plt.show()
while True:                                     # infinite loop through frames
    try:
        for i, vm in enumerate(list_vm):
            progressbar(i/len(list_vm))
            fig, ax = plot(vm, fig=fig, ax=ax)  # update plot
    except WindowClosedException:
        break                                   # stop when window is closed
plt.ioff()                                      # disable interactive mode
progressbar(1)

# compute times
list_t = np.array(list(map(lambda vm: vm.time, list_vm)))                       # times of each frame
np.testing.assert_almost_equal(np.diff(list_t).std()/np.diff(list_t).mean(), 0) # check frames are linearly spaced
lag_time = np.diff(list_t).mean()                                               # time between frames

# compute positions
cells = list_vm[0].getVertexIndicesByType("centre")                 # indices of cell centres (from first frame)
positions = np.array(list(map(
    lambda vm: itemgetter(*cells)(vm.getPositions(wrapped=False)),  # unwrapped positions of centres
    list_vm)))

# compute mean squared displacement
msd = []
for t in range(1, len(list_t)):
    disp = np.roll(positions, -t, axis=0)[:-t] - positions[:-t] # compute displacements over t frames
    disp = disp - disp.mean(axis=-2, keepdims=True)             # remove centre of mass displacement
    msd += [[t*lag_time, (disp**2).sum(axis=-1).mean()]]        # mean squared displacement
msd = np.array(msd)
with open("msd.p", "wb") as dump:
    pickle.dump(msd, dump)

# plot mean squared displacement
fig, ax = plt.subplots()
ax.set_xlabel(r"$t$")
ax.set_xscale("log")
ax.set_ylabel(r"$\mathrm{MSD}(t)$")
ax.set_yscale("log")
ax.plot(msd[:, 0], msd[:, 1], marker="s")
plt.show()

