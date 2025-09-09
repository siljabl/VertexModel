import time
import argparse
import subprocess


import numpy as np


from pathlib  import Path
from datetime import datetime

from cells.exponents import float_to_letters

from utils.config_functions import *

# Define paths
config_path = "data/simulated/configs/"
output_path = "data/simulated/raw/"
movies_path = "data/simulated/videos/"



def create_ouput_directory(script, config, seed, prefix=None):
    """
    Generates standard name and creates directory
    """
    
    # Number of cells in simulation
    Ngrid  = get_value(config, 'Nvertices')
    Ncells = Ngrid ** 2 / 3

    # Streching/compression of cells
    rho = get_value(config, 'rho')

    # Name on directory
    directory = f"{Path(script).stem}_N{int(Ncells)}_rho{int(100*rho)}_seed{seed}"

    # Create folders
    Path(f"{config_path}{directory}/").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}{directory}/").mkdir(parents=True, exist_ok=True)
    Path(f"{movies_path}{directory}/").mkdir(parents=True, exist_ok=True)

    return directory


def generate_seed(digits):
    """ Generates random seed below 1e<digits> """
    return int(time.time()) % 10 ** digits



if __name__ == "__main__":

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run several runs")
    parser.add_argument('script',         type=str,  help='Simulation script')
    parser.add_argument('-N', '--nruns',  type=int,  help="Number of runs to so", default=2)
    parser.add_argument('-s', '--seed',   type=int,  help="Simulation seed",      default=None)
    parser.add_argument('-c', '--config', type=str,  help='Path to config file',  default='data/simulated/configs/config.json')
    parser.add_argument('-p', '--params', nargs='*', help='Additional parameters in the form key_value')
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)

    # Set simulation seed
    if args.seed == None:
        args.seed = generate_seed(3)
    np.random.seed(args.seed)

    # Create subfolder for ensemble
    output_dir = create_ouput_directory(args.script, config, args.seed)

    # Define run-specific seeds
    VMseeds = np.random.randint(1e3, size=10)      # vertex model object seed
    Vseeds  = np.random.randint(1e3, size=10)      # volume distribution seed

    for run in range(args.nruns):

        # Update config
        update_value(config, 'VMseed', VMseeds[run])
        update_value(config, 'Vseed',  Vseeds[run])

        # Save with updated seeds
        save_config(args.config, config)

        # Prepare the command to run the simulation
        command = [
            'python', 
            args.script,
            '--dir', output_dir,
            '--config', args.config
        ]
        subprocess.run(command, check=True)
        
    print(f"All simulations completed. Results saved in: {output_dir}")
