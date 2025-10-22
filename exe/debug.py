import os
import sys
import pickle
import traceback
import subprocess
import numpy as np
import scipy as sc
from tempfile import mkdtemp

from cells.plot import plot
from cells.bind import VertexModel
from cells.init import movie_sh_fname



def main():

    # PARAMETERS

    # Lattice
    seed  = 0                                               # random number generator seed
    Ngrid = 30                                              # number of vertices in each dimension. Ncell = Ngrid**2 / 3
    Lgrid = 64                                              # length of lattice
    rgrid = Lgrid / Ngrid                                   # length scale of triangular lattice

    # Cell
    rhex  = 1                                               # reference side lenght of regular cell (hexagon)
    A0    = (3**(3/2) / 2) * (rgrid)**2                     # initial cell area
    V0    = (3**2 / 2) * rhex**3                            # cell volume
    stdV0 = 0.4 * V0                                        # standard deviation of cell volume distribution
    Vmin  = 0.6 * V0                                        # lower limit on volume
    Vmax  = 3.5 * V0                                        # upper limit on volume

    # Forces
    Lambda = 1                                              # surface tension
    tauV   = 0                                              # inverse increase rate in V0 unit
    v0     = 1                                              # self-propulsion velocity
    taup   = 3                                              # self-propulsion persistence time
    eta    = 1                                              # vertex-vertex pair drag coefficient

    # Integration
    dt      = 0.1                                           # integration time step
    delta   = 0.02                                          # length below which T1s are triggered
    epsilon = 0.002                                         # edges have length delta+epsilon after T1s
    period  = 100                                           # period between frames
    Nframes = 10                                            # number of frames in simulation

    # INITIALISATION

    # Set seed
    np.random.seed(seed)

    # Vertex model object
    vm = VertexModel(np.random.randint(1e5))                        # initialise vertex model object
    vm.initRegularTriangularLattice(size=Ngrid, hexagonArea=A0)     # initialise periodic system

    # Add forces
    vm.addActiveBrownianForce("abp", v0, taup)                      # centre active Brownian force
    vm.addSurfaceForce("surface", Lambda, V0, tauV)                 # surface tension force
    vm.setPairFrictionIntegrator(eta)                               # add pair dissipation

    vm.vertexForces["surface"].volume = dict(map(                   # set cell volume
        lambda i: (i, sc.stats.truncnorm((Vmin-V0)/stdV0, (Vmax-V0)/stdV0, loc=V0, scale=stdV0).rvs()),
        vm.vertexForces["surface"].volume))



    # SIMULATION
    # frames directory
    _frames_dir = mkdtemp()
    print("Save frames to temp directory \"%s\"." % _frames_dir, file=sys.stderr)
    index = 0

    # outputs
    with open("out.p", "wb") as dump: pass
    fig, ax = plot(vm, fig=None, ax=None)                   # initialise plot with first frame

    # simulation
    frame = 0
    for step in range(0, Nframes):
        # output is appended to file
        with open(f"out.p", "ab") as dump: pickle.dump(vm, dump)

        # plot snapshot
        plot(vm, fig=fig, ax=ax, update=True)

        # save frame
        while True:
            try:
                fig.savefig(os.path.join(_frames_dir, "%05d.png" % index))
                break
            
            except SyntaxError:
                # dirty fix to "SyntaxError: not a PNG file" with multiple matplotlib instances
                print(traceback.format_exc(), file=sys.stderr)
                pass
        frame += 1

        # integrate
        vm.nintegrate(period, dt, delta, epsilon)


    # make movie
    subprocess.call([movie_sh_fname,
    "-d", _frames_dir, "-p", sys.executable, "-y"])

if __name__ == "__main__":
    main()

