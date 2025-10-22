import os
import sys
import shutil
import pickle
import argparse
import platform
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

from run_ensemble import create_dirname

import matplotlib
matplotlib.use("Agg")


# Define paths
#config_path = "data/simulated/configs/"
output_path = "data/simulated/raw/"


if platform.node() != 'silja-work':
    #config_path = "../../../../hdd_data/silja/VertexModel_data/simulated/configs/"
    output_path = "../../../../hdd_data/silja/VertexModel_data/simulated/raw/"
                       


def create_filename(config_file, ensemble=False):

    seed = get_value(config_file, 'seed')

    if ensemble:
        filename = f"{Path(__file__).stem}_seed{seed}"
    else:
        filename = create_dirname(f"{Path(__file__).stem}.py", config_file, filename=True)

    return filename



def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run simulation constant cell volume and active brownian motion")
    parser.add_argument('-d', '--dir',    type=str,  help='Save in subfolders data/*/dir/. Creates dir if not existing.', default='')
    parser.add_argument('-c', '--config', type=str,  help='Path to config file',                       default='data/simulated/configs/config_nodivision_pairdissipation.json')
    parser.add_argument('-p', '--params', nargs='*', help='Additional parameters in the form key_value')
    parser.add_argument('--cbar0',        type=str,  help='How define 0 level of cbar in vm video',    default='absolute')
    parser.add_argument('--frames_dir',   type=str,  help='Where to save frames',    default='../../../../hdd_data/silja/VertexModel_data/simulated/frames/')
    parser.add_argument('--ensemble',                help='Defines whether run is part of ensemble execution', action='store_true')
    args = parser.parse_args()



    # CONFIG

    # Load config file
    config_path = args.config
    config_file = load_config(config_path)
    
    # Add script and date
    config_file["script"] = __file__
    config_file["date"]   = datetime.now().strftime('%Y%m%d_%H%M%S')

    # If additional parameters were provided, update the config
    if args.params:
        # Ensure we have an even number of parameters
        if len(args.params) % 2 != 0:
            raise ValueError("Parameters should be provided in pairs of key value.")
        
        # Iterate over the arguments in pairs
        for i in range(0, len(args.params), 2):
            key   = args.params[i]
            value = args.params[i + 1]
            
            # Update the config dictionary
            update_value(config_file, key, value)



    # LOAD PARAMETERS

    config = config_file                                    # for readability                             

    # Lattice
    seed  = config['simulation']['seed']                    # random number generator seed
    Ngrid = config['simulation']['Nvertices']               # number of vertices in each dimension. Ncell = Ngrid**2 / 3
    Lgrid = config['simulation']['Lgrid']                   # length of lattice
    rgrid = Lgrid / Ngrid                                   # length scale of triangular lattice

    # Cell
    rhex  = config['physics']['rhex']                       # reference side lenght of regular cell (hexagon)
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
    eta    = config['physics']['eta']                       # vertex-vertex pair drag coefficient

    # Integration
    dt      = config['simulation']['dt']                    # integration time step
    delta   = config['simulation']['delta']                 # length below which T1s are triggered
    epsilon = config['simulation']['epsilon']               # edges have length delta+epsilon after T1s
    period  = config['simulation']['period']                # period between frames
    Nframes = config['simulation']['Nframes']               # number of frames in simulation



    rho = cell_density(Ngrid, Lgrid)
    update_value(config_file, 'rho', rho)
    
    # Save simulation-specific config file
    fname = create_filename(config_file, args.ensemble)
    print("Simulation name: ", fname)


    # DEFINE PATHS

    # Check if subfolders exists, if not create
    if args.dir != '':
        args.dir = f"{args.dir}/"
    path_to_config = f"{Path(config_path).parent}/{args.dir}"
    path_to_output = f"{output_path}/{args.dir}"
    path_to_frames = f"{args.frames_dir}/{args.dir}/{fname}"

    Path(path_to_config).mkdir(parents=True, exist_ok=True)
    Path(path_to_output).mkdir(parents=True, exist_ok=True)
    Path(path_to_frames).mkdir(parents=True, exist_ok=True)

       
    # Save frames in temporary directory
    print("Save frames to temp directory \"%s\"." % path_to_frames, file=sys.stderr)

    save_config(f"{path_to_config}{fname}.json", config_file)



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

    cbar_zero = args.cbar0

    # outputs
    with open(f"{path_to_output}{fname}.p", "wb") as dump: pass     # output file is created
    fig, ax = plot(vm, fig=None, ax=None, cbar_zero=cbar_zero)                           # initialise plot with first frame


    # simulation
    frame = 0
    for step in range(0, Nframes):
        # output is appended to file
        with open(f"{path_to_output}{fname}.p", "ab") as dump: pickle.dump(vm, dump)

        # plot snapshot
        save_snapshot(vm, fig, ax, path_to_frames, frame, cbar_zero=cbar_zero)
        frame += 1

        # integrate
        vm.nintegrate(period, dt, delta, epsilon)


    # # make movie
    # subprocess.call([movie_sh_fname,
    #                 "-d", path_to_frames,
    #                 "-o", f"{path_to_frames}{fname}.mp4",
    #                 "-p", sys.executable,
    #                 "-y"])
    
    os.system('stty sane')
    # shutil.rmtree(path_to_frames)

if __name__ == "__main__":
    main()
