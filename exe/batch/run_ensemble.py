import os
import sys
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
from multiprocessing import Pool

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))


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
    parser.add_argument('script',         type=str,  help='Simulation script')
    parser.add_argument('-N', '--Nruns',  type=int,  help="Number of runs to so",         default=4)
    parser.add_argument('-P', '--Npool',  type=int,  help="Number of parallel processes", default=16)
    parser.add_argument('-s', '--seed',   type=int,  help="Simulation seed",              default=None)
    parser.add_argument('-c', '--config', type=str,  help='Path to config file',          default='configs/base/config_constant_volume.json')
    args = parser.parse_args()


    # Set simulation seed
    if args.seed == None:
        args.seed = generate_seed(3)
    np.random.seed(args.seed)


    # Prepare the commands for each run
    commands = []
    for run in range(args.Nruns):
        command = [
            'python', 
            args.script,
            '--config', args.config,
            '--seed', str(np.random.randint(1e3)),
        ]
        commands.append(command)

    # Use multiprocessing to run the simulations in parallel
    with Pool(processes=args.Npool) as pool:

        # Execute the list of commands in parallel
        pool.map(run_simulation, commands)

    print(f"All simulations completed.")


if __name__ == "__main__":
    main()
