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


# Define paths
config_path = "data/simulated/configs/"
output_path = "data/simulated/raw/"
movies_path = "data/simulated/videos/"
object_path = "data/simulated/obj/"


def create_ouput_directory(script, config_file, seed, prefix=None):
    """
    Generates standard name and creates directory
    """
    
    # Number of cells in simulation
    Ngrid  = get_value(config_file, 'Nvertices')
    Ncells = Ngrid ** 2 / 3

    # Streching/compression of cells
    rho = get_value(config_file, 'rho')

    # Name on directory
    timestamp = datetime.now().strftime('%Y%m%d')
    directory = f"{Path(script).stem}_{timestamp}_seed{seed}"
    #directory = f"{Path(script).stem}_N{int(Ncells)}_rho{int(100*rho)}_seed{seed}"

    # Create folders
    Path(f"{config_path}{directory}/").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}{directory}/").mkdir(parents=True, exist_ok=True)
    Path(f"{movies_path}{directory}/").mkdir(parents=True, exist_ok=True)
    Path(f"{object_path}{directory}/").mkdir(parents=True, exist_ok=True)
    Path(f"{object_path}{directory}/averages").mkdir(parents=True, exist_ok=True)

    return directory


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
    parser.add_argument('-N', '--Nruns',  type=int,  help="Number of runs to so",         default=2)
    parser.add_argument('-P', '--npool',  type=int,  help="Number of parallel processes", default=16)
    parser.add_argument('-s', '--seed',   type=int,  help="Simulation seed",              default=None)
    parser.add_argument('-c', '--config', type=str,  help='Path to config file',  default='data/simulated/configs/config.json')
    parser.add_argument('-p', '--params', nargs='*', help='Additional parameters in the form key_value')
    args = parser.parse_args()

    # Load configurations
    config_file = load_config(args.config)

    # Set simulation seed
    if args.seed == None:
        args.seed = generate_seed(3)
    np.random.seed(args.seed)

    # Create subfolder for ensemble
    output_dir = create_ouput_directory(args.script, config_file, args.seed)

    # Prepare the commands for each run
    commands = []
    for run in range(args.Nruns):
        command = [
            'python', 
            args.script,
            '--dir',    output_dir,
            '--config', args.config,
            '--run_id', str(run),
            '--params', 'seed', str(np.random.randint(1e3))
        ]
        commands.append(command)

    # Use multiprocessing to run the simulations in parallel
    with Pool(processes=args.npool) as pool:

        # Execute the list of commands in parallel
        pool.map(run_simulation, commands)

    print(f"All simulations completed. Results saved in: {output_dir}")


    # Load random config
    path_to_random_run = glob.glob(f"{config_path}{output_dir}/*.json")[0]
    config_file = load_config(path_to_random_run)
    
    # Update with ensemble seed
    update_value(config_file, key='seed', val=args.seed)
    
    # Add number of runs/states in ensemble
    config_file['Nruns'] = args.Nruns

    # Save and delete folder
    save_config(f"{config_path}{output_dir}.json", config_file)
    shutil.rmtree(f"{config_path}{output_dir}")

if __name__ == "__main__":
    main()