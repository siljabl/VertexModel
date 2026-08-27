import os
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import pickle
import argparse
import numpy as np
import utils.config_functions as cfg
import utils.vm_integration as integrate

from datetime import datetime
from cells.bind import VertexModel
from paths_config import SIM_RAW_DIR, SIM_FRAMES_DIR
from utils.path_handling import create_run_fname
from utils.exception_handlers import save_frame
from utils.vm_plotting import plot_frame
from utils.vm_setup import initalise_vm_lattice, set_cell_volumes, initialise_vm_forces





def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run simulation constant cell volume and active brownian motion")
    parser.add_argument('-c', '--config_path', type=str,  help='Path to config file',                       default='configs/base/config_constant_volume.json')
    parser.add_argument('-s', '--seed',        type=int,  help='Simulation seed',                           default=None)
    parser.add_argument('--cbar0',             type=str,  help='How define 0 level of cbar in vm video',    default='absolute')
    parser.add_argument('--init_time',         type=int,  help='Number of initialisation frames',           default=100)
    args = parser.parse_args()


    # Load config file
    config = cfg.load(args.config_path)
    config["date"] = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create simulation run filename
    if args.seed:
        config['simulation']['seed'] = args.seed
    seed  = config['simulation']['seed']

    dirname, runname = create_run_fname(config)
    fname = f"{Path(__file__).stem}/{dirname}/{runname}"

    # configs
    cfg_dir = Path("configs") / Path(__file__).stem / dirname
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg_dir / f"{runname}.json", config, script_file=__file__)

    # raw data
    raw_dir = Path(SIM_RAW_DIR) / Path(__file__).stem / dirname
    raw_dir.mkdir(parents=True, exist_ok=True)

    # frames
    frames_dir = Path(SIM_FRAMES_DIR) / fname
    frames_dir.mkdir(parents=True, exist_ok=True)



    # INITIALISATION

    # Set seed of simulation
    np.random.seed(seed)

    vm = VertexModel(np.random.randint(1e5))
    vm = initalise_vm_lattice(vm, config)
    vm = initialise_vm_forces(vm, config)
    vm = set_cell_volumes(vm, config)



    # SIMULATION

    # outputs
    with open(raw_dir / f"{runname}.p", "wb") as dump: pass      # output file is created
    fig, ax = plot_frame(vm, fig=None, ax=None, cbar_zero=args.cbar0)      # initialise plot with first frame

    # simulation
    frame = 0
    for step in range(0, config['simulation']['Nframes']):
        # output is appended to file
        with open(raw_dir / f"{runname}.p", "ab") as dump: pickle.dump(vm, dump)

        # plot snapshot
        if frame > args.init_time:
            save_frame(vm, fig, ax, f"{SIM_FRAMES_DIR}/{fname}", frame, cbar_zero=args.cbar0)
        frame += 1

        # integrate
        for i in range(config['simulation']['period']):
            integrate.one_timestep(vm, config)
   
    os.system('stty sane')

    print(f"Output saved in {raw_dir}/{runname}")


if __name__ == "__main__":
    main()
