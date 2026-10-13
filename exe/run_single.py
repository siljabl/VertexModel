import os
import sys
from pathlib import Path

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import pickle
import argparse
import numpy as np
import utils.config_io as cfg
import utils.vm_integration as integrate

from datetime import datetime
from cells.bind import VertexModel
from paths_config import SIM_RAW_DIR, SIM_FRAMES_DIR
from utils.vm_plotting import plot_frame, save_frame
from utils.vm_setup import initialise_vm_lattice, set_cell_volumes, initialise_vm_forces
from utils.vm_observables import centre_indices



def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run simulation constant cell volume and active brownian motion")
    parser.add_argument('mode',                type=str,                               help='Simulation mode (constant_volume, growing_volume, death_and_division)')
    parser.add_argument('-c', '--config_path', type=str,  default='configs/base.json', help='Path to config file')
    parser.add_argument('-s', '--seed',        type=int,  default=None,                help='Simulation seed')
    parser.add_argument('--cbar0',             type=str,  default='absolute',          help='How define 0 level of cbar in vm video')
    args = parser.parse_args()

    assert args.mode in ("constant_volume", "growing_volume", "death_and_division", "emergent_volume")

    # Load config file
    config = cfg.load(args.config_path)
    config["date"] = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set the simulation seed
    if args.seed:
        config['simulation']['seed'] = args.seed
    seed = config['simulation']['seed']

    # Create simulation run filename
    dirname, runname = cfg.create_run_fname(config, args.mode)
    fname = f"{args.mode}/{dirname}/{runname}"

    # Creare folders for output
    # configs
    cfg_dir = Path("configs") / args.mode / dirname
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(cfg_dir / f"{runname}.json", config, mode=args.mode)

    # raw data
    raw_dir = Path(SIM_RAW_DIR) / args.mode / dirname
    raw_dir.mkdir(parents=True, exist_ok=True)

    # frames
    frames_dir = Path(SIM_FRAMES_DIR) / fname
    frames_dir.mkdir(parents=True, exist_ok=True)



    # INITIALISATION

    # seed
    np.random.seed(seed)

    vm = VertexModel(np.random.randint(1e5))
    vm = initialise_vm_lattice(vm, config)
    vm = initialise_vm_forces(vm, config)
    vm = set_cell_volumes(vm, config, distribution=config["cells"]["distribution"])



    # SIMULATION

    # outputs
    with open(raw_dir / f"{runname}.p", "wb") as dump: pass                # output file is created
    fig, ax = plot_frame(vm, fig=None, ax=None, cbar_zero=args.cbar0)      # initialise plot with first frame

    # frames
    period       = config['integration']['period']
    n_transient  = config['integration']['n_transient']
    n_production = config['integration']['n_production']
    
    frame    = 0
    n_frames = n_production + n_transient

    for step in range(0, n_frames):
        # output is appended to file
        with open(raw_dir / f"{runname}.p", "ab") as dump: pickle.dump(vm, dump)

        # plot and save snapshot
        if frame > n_transient:
            save_frame(vm, fig, ax, f"{SIM_FRAMES_DIR}/{fname}", frame, cbar_zero=args.cbar0)


        # integration
        for i in range(period):
            integrate.one_timestep(vm, config)


            # Growing volume: cell growth at every time step, cell death during transient state
            if args.mode == "growing_volume":
                integrate.cell_growth(vm, config)

                # letting cells die during transient state
                if frame < n_transient and i == int(period / 2):
                    integrate.cell_death(vm, config)


            # Death and division: cell growth at every time step, cell death at every period'th time step
            elif args.mode == "death_and_division":
                integrate.cell_growth(vm, config)

                # letting cells die during transient state
                if i == int(period / 2):
                    integrate.cell_death(vm, config)


            # Emergent volume: cell growth and cell death only during transient state
            elif args.mode == "emergent_volume":
                # letting cells grow during transient state
                if frame < n_transient:
                    integrate.cell_growth(vm, config)
                    
                    # letting cells die during transient state
                    if i == int(period / 2):
                        integrate.cell_death(vm, config)


        # division
        if args.mode in ("growing_volume", "death_and_division"):
            integrate.cell_division(vm, config)

        # division
        elif args.mode == "emergent_volume" and frame < n_transient:
            integrate.cell_division(vm, config)

        # print cell count
        print("frame: ", frame, ", rho: ", int(10**6*len(centre_indices(vm)) / (600*520)))

        frame += 1
   
    os.system('stty sane')

    print(f"Output saved in {raw_dir}/{runname}")


if __name__ == "__main__":
    main()
