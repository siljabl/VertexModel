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
from cells.init import movie_sh_fname

from utils.config_functions   import *
from utils.vm_functions       import *
from utils.plotting_functions import plot
from utils.exception_handlers import save_snapshot

import matplotlib
matplotlib.use("Agg")


def filename(config_file, sufix=''):
        # Number of cells in simulation
    Ngrid  = get_value(config_file, 'Nvertices')
    Ncells = Ngrid ** 2 / 3

    # Streching/compression of cells
    rho = get_value(config_file, 'rho')

    # Name on directory
    filename = f"{Path(__file__).stem}_N{int(Ncells)}_rho{int(100*rho)}_{sufix}"

    return filename



def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run simulation constant cell volume and active brownian motion")
    parser.add_argument('-d', '--dir',    type=str,  help='Save in subfolders data/*/dir/. Creates dir if not existing.', default='')
    parser.add_argument('-c', '--config', type=str,  help='Path to config file',                       default='data/simulated/configs/config.json')
    parser.add_argument('-i', '--run_id', type=int,  help='Identity to separate parallel runs',        default=None)
    parser.add_argument('-p', '--params', nargs='*', help='Additional parameters in the form key_value')
    args = parser.parse_args()



    # DEFINE PATHS

    # Check if subfolders exists, if not create
    if args.dir != '':
        args.dir = f"{args.dir}/"
    path_to_config = f"data/simulated/configs/{args.dir}"
    path_to_output = f"data/simulated/raw/{args.dir}"
    path_to_movies = f"data/simulated/videos/{args.dir}"

    Path(path_to_config).mkdir(parents=True, exist_ok=True)
    Path(path_to_output).mkdir(parents=True, exist_ok=True)
    Path(path_to_movies).mkdir(parents=True, exist_ok=True)

       
    # Save frames in temporary directory
    _frames_dir = mkdtemp()
    print("Save frames to temp directory \"%s\"." % _frames_dir, file=sys.stderr)



    # CONFIG

    # Load config file
    config_path = args.config
    config_file = load_config(config_path)

    # If additional parameters were provided, update the config
    if args.params:
        # Ensure we have an even number of parameters
        if len(args.params) % 2 != 0:
            raise ValueError("Parameters should be provided in pairs of key value.")
        
        # Iterate over the arguments in pairs
        for i in range(0, len(args.params), 2):
            key   = args.params[i]
            value = float(args.params[i + 1])
            
            # Update the config dictionary
            update_value(config_file, key, value)


    # Use script name and timing as name on output
    if args.run_id == None:
        sufix = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        sufix = f"run{args.run_id}"

    fname = filename(config_file, sufix)
    print("Simulation name: ", fname)


    # Save simulation-specific config file
    save_config(f"{path_to_config}{fname}.json", config_file)



    # LOAD PARAMETERS

    config = config_file                                    # for readability                             

    # Define cell
    rhex  = config['physics']['rhex']                       # reference side lenght of regular cell (hexagon)
    rho   = config['physics']['rho']                        # defines compression/stretching of cells

    # Lattice
    seed  = config['simulation']['seed']                    # random number generator seed
    Ngrid = config['simulation']['Nvertices']               # number of vertices in each dimension. Ncell = Ngrid**2 / 3
    rgrid = rhex / rho                                      # length scale of triangular lattice

    # Cell distributions
    A0    = hexagon_area(rgrid)                             # initial cell area
    V0    = hexagon_volume(rhex)                            # cell volume
    stdV0 = config['experimental']['stdV0'] * V0            # standard deviation of cell volume distribution
    Vmin  = config['experimental']['Vmin']  * V0            # lower limit on volume
    Vmax  = config['experimental']['Vmax']  * V0            # upper limit on volume

    # Forces
    Lambda = config['physics']['Lambda']                    # surface tension
    tauV   = config['physics']['tauV']                      # inverse increase rate in V0 unit
    v0     = config['physics']['v0']                        # self-propulsion velocity
    taup   = config['physics']['taup']                      # self-propulsion persistence time

    # Integration
    dt      = config['simulation']['dt']                    # integration time step
    delta   = config['simulation']['delta']                 # length below which T1s are triggered
    epsilon = config['simulation']['epsilon']               # edges have length delta+epsilon after T1s
    period  = config['simulation']['period']                # period between frames
    Nframes = config['simulation']['Nframes']               # number of frames in simulation



    # INITIALISATION

    # Set seed
    np.random.seed(seed)

    # Vertex model object
    vm = VertexModel(np.random.randint(1e5))                        # initialise vertex model object
    vm.initRegularTriangularLattice(size=Ngrid, hexagonArea=A0)     # initialise periodic system


    # Add forces
    vm.addActiveBrownianForce("abp", v0, taup)                      # centre active Brownian force
    vm.addSurfaceForce("surface", Lambda, V0, tauV)                 # surface tension force

    vm.vertexForces["surface"].volume = dict(map(                   # set cell volume
        lambda i: (i, sc.stats.truncnorm((Vmin-V0)/stdV0, (Vmax-V0)/stdV0, loc=V0, scale=stdV0).rvs()),
        vm.vertexForces["surface"].volume))



    # SIMULATION

    # outputs
    with open(f"{path_to_output}{fname}.p", "wb") as dump: pass     # output file is created
    fig, ax = plot(vm, fig=None, ax=None, cbar_zero='average')                           # initialise plot with first frame


    # simulation
    frame = 0
    for step in range(0, Nframes):
        # output is appended to file
        with open(f"{path_to_output}{fname}.p", "ab") as dump: pickle.dump(vm, dump)

        # plot snapshot
        save_snapshot(vm, fig, ax, _frames_dir, frame, cbar_zero='average')
        frame += 1

        # integrate
        vm.nintegrate(period, dt, delta, epsilon)


    # make movie
    subprocess.call([movie_sh_fname,
                    "-d", _frames_dir,
                    "-o", f"{path_to_movies}{fname}.mp4",
                    "-p", sys.executable,
                    "-y"])


if __name__ == "__main__":
    main()

