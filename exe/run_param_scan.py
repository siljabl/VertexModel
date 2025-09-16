""" Dummy script that might be included later """
import time
import glob
import shutil
import argparse
import subprocess
import numpy as np

from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
from utils.config_functions import *

from run_ensemble import create_dirname


def generate_seed(digits):
    """ Generates random seed below 1e<digits> """
    return int(time.time()) % 10 ** digits


def run_simulation(command):
    """ Runs a single simulation command. """
    result = subprocess.run(command, check=True)
    return result


def main():

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run several runs")
    parser.add_argument('script',         type=str,   help='Simulation script')
    parser.add_argument('param',          type=str,   help="Parameter to scan")
    parser.add_argument('-l', '--list',   nargs='*',  help="List of parameter values",                  default=None)
    parser.add_argument('-r', '--range',  nargs='*',  help="Linear parameter range (min, max, Nparam)", default=None)
    parser.add_argument('-s', '--seed',   type=int,   help="Simulation seed",                           default=None)
    parser.add_argument('-N', '--Nruns',  type=int,   help="Number of runs to do in ensemble",          default=2)
    parser.add_argument('-P', '--Npool',  type=int,   help="Number of parallel processes",              default=16)
    parser.add_argument('-c', '--config', type=str,   help='Path to config file', default='data/simulated/configs/config_nodivision.json')
    parser.add_argument('--ensemble',                 help='Defines whether run is part of ensemble execution', action='store_true')
    args = parser.parse_args()

    assert args.range != None or args.list != None, f"Must provide either range or list of values for {args.param}"

    if args.list == None:
        assert len(args.range) == 3, f"Wrong format on input '--range'. Should be '-r <min> <max> <Nparam>'"
        param_range = np.linspace(args.range[0], args.range[1], args.range[2])
    else:
        param_range = [param for param in args.list]
    
    Nparam = len(param_range)

    # Set simulation seed
    if args.seed == None:
        args.seed = generate_seed(3)
    np.random.seed(args.seed)


    if args.ensemble:

        # Prepare the commands for each run
        commands = []
        for run, param in zip(range(Nparam),param_range):

            # Update config
            config_file = load_config(args.config)
            update_value(config_file, key=args.param, val=param)
            save_config(args.config, config_file)

            command = [
                'python', 
                'exe/run_ensemble.py',
                args.script,
                '--Nruns',  str(args.Nruns),
                '--Npool',  str(args.Npool),
                '--seed',   str(np.random.randint(1e3)),
                '--config', args.config
            ]

            run_simulation(command)

    else:
        
        # Prepare the commands for each run
        commands = []
        for run, param in zip(range(Nparam), param_range):
            command = [
                'python',
                args.script,
                '--config', args.config,
                '--params', 'seed',     str(np.random.randint(1e3)),
                            args.param, param
            ]
            commands.append(command)

        # Use multiprocessing to run the simulations in parallel
        with Pool(processes=args.Npool) as pool:
            pool.map(run_simulation, commands)


if __name__ == "__main__":
    main()