import time
import glob
import shutil
import argparse
import platform
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

if platform.node() != 'silja-work':
    print(f"Running simulation from {platform.node()}")
    config_path = "../../../../hdd_data/silja/VertexModel_data/simulated/configs/"
    output_path = "../../../../hdd_data/silja/VertexModel_data/simulated/raw/"
    #movies_path = "../../../../hdd_data/silja/VertexModel_data/simulated/videos/"
    print(f"Saving output in {output_path}")


def create_dirname(script, config_file, filename=False):
    """
    Generates standard name and creates directory
    """
    
    # Simulation parameters
    Ngrid = get_value(config_file, 'Nvertices')
    Lgrid = get_value(config_file, 'Lgrid')
    gamma = get_value(config_file, 'gamma')
    v0    = get_value(config_file, 'v0')
    taup  = get_value(config_file, 'taup')
    eta   = get_value(config_file, 'eta')

    # Transform float to int
    gamma = int(gamma)
    v0    = int(v0)
    taup  = int(100*taup)
    eta   = int(100*eta)

    # Name on directory
    timestamp = datetime.now().strftime('%Y%m%d')
    directory = f"{Path(script).stem}_{timestamp}_N{Ngrid}_L{Lgrid}_gamma{gamma}_v0{v0}_taup{taup}_eta{eta}"


    if not filename:
        # Create folders
        Path(f"{config_path}{directory}/").mkdir(parents=True, exist_ok=True)
        Path(f"{output_path}{directory}/").mkdir(parents=True, exist_ok=True)
        #Path(f"{movies_path}{directory}/").mkdir(parents=True, exist_ok=True)

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
    parser.add_argument('-N', '--Nruns',  type=int,  help="Number of runs to so",         default=10)
    parser.add_argument('-P', '--Npool',  type=int,  help="Number of parallel processes", default=16)
    parser.add_argument('-s', '--seed',   type=int,  help="Simulation seed",              default=None)
    parser.add_argument('-c', '--config', type=str,  help='Path to config file',          default='../../../../hdd_data/silja/VertexModel_data/simulated/configs/config.json')
    
    parser.add_argument('--pair_dissipation', action="store_true", help="Adding pair-dissipation.")
    args = parser.parse_args()

    # Load configurations
    config_file = load_config(args.config)

    # Set simulation seed
    if args.seed == None:
        args.seed = generate_seed(3)
    np.random.seed(args.seed)

    # Create subfolder for ensemble
    output_dir = create_dirname(args.script, config_file)

    # Prepare the commands for each run
    commands = []
    for run in range(args.Nruns):
        command = [
            'python', 
            args.script,
            '--dir',    output_dir,
            '--config', args.config,
            '--params', 'seed', str(np.random.randint(1e3)),
            '--ensemble'
        ]
        commands.append(command)

    # Use multiprocessing to run the simulations in parallel
    with Pool(processes=args.Npool) as pool:

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
