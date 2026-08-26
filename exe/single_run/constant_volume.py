import os, sys
from pathlib  import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import pickle
import argparse
import numpy as np
import utils.config_functions as config

from datetime                 import datetime
from cells.bind               import VertexModel
from paths_config             import SIM_RAW_DIR, SIM_FRAMES_DIR
from utils.path_handling      import create_run_fname
from utils.plotting_functions import plot
from utils.exception_handlers import save_snapshot
from utils.vm_setup           import initalise_vm_lattice, set_cell_volumes, initialise_vm_forces




def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run simulation constant cell volume and active brownian motion")
    parser.add_argument('-d', '--dir',         type=str,  help='Save in subfolders data/*/dir/. Creates dir if not existing.', default='')
    parser.add_argument('-c', '--config_path', type=str,  help='Path to config file',                       default='configs/base/config_constant_volume.json')
    parser.add_argument('--cbar0',             type=str,  help='How define 0 level of cbar in vm video',    default='absolute')
    parser.add_argument('--init_time',         type=int,  help='Number of initialisation frames',           default=100)
    args = parser.parse_args()


    # Load config file
    config_dict = config.load(args.config_path)
    config_dict["date"] = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create simulation run filename
    seed  = config_dict['simulation']['seed']

    fname = create_run_fname(config_dict, __file__)

    config.save(f"configs/{Path(__file__).stem}/{fname}.json", config_dict, script_file=__file__)



    # INITIALISATION

    # Set seed of simulation
    np.random.seed(seed)

    vm = VertexModel(np.random.randint(1e5))
    vm = initalise_vm_lattice(vm, config_dict)
    vm = initialise_vm_forces(vm, config_dict)
    vm = set_cell_volumes(vm, config_dict)



    # SIMULATION

    # outputs
    with open(f"{SIM_RAW_DIR}/{fname}.p", "wb") as dump: pass      # output file is created
    fig, ax = plot(vm, fig=None, ax=None, cbar_zero=args.cbar0)      # initialise plot with first frame

    # simulation
    frame = 0
    for step in range(0, config_dict['simulation']['Nframes']):
        # output is appended to file
        with open(f"{SIM_RAW_DIR}/{fname}.p", "ab") as dump: pickle.dump(vm, dump)

        # plot snapshot
        if frame > args.init_time:
            save_snapshot(vm, fig, ax, f"{SIM_FRAMES_DIR}/{fname}", frame, cbar_zero=args.cbar0)
        frame += 1

        # integrate
        vm.nintegrate(config_dict['simulation']['period'], 
                      config_dict['simulation']['dt'], 
                      config_dict['simulation']['delta'], 
                      config_dict['simulation']['epsilon'])

   
    os.system('stty sane')


if __name__ == "__main__":
    main()
