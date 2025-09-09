import argparse
import subprocess

from pathlib  import Path
from datetime import datetime

from cells.exponents import float_to_letters

from utils.config_functions import *

# Define paths
config_path = "data/simulated/configs/"
output_path = "data/simulated/raw/"
movies_path = "data/simulated/videos/"



def create_ouput_directory(script, config, prefix=None):
    """
    Generates standard name and creates directory
    """
    
    # Number of cells in simulation
    Ngrid  = get_value(config, 'Nvertices')
    Ncells = Ngrid ** 2 / 3

    # Streching/compression of cells
    rho = get_value(config, 'rho')

    # Seed/stamp/identifier

    # Name on directory
    directory = f"{Path(script).stem}_N{int(Ncells)}_rho{int(100*rho)}"    # add seed/identifier

    # Create folders
    Path(f"{config_path}{directory}/").mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}{directory}/").mkdir(parents=True, exist_ok=True)
    Path(f"{movies_path}{directory}/").mkdir(parents=True, exist_ok=True)

    return directory



if __name__ == "__main__":

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run several runs")
    parser.add_argument('script',         type=str,  help='Simulation script')
    parser.add_argument('-N', '--nruns',  type=int,  help="Number of runs to so", default=2)
    parser.add_argument('-s', '--seed',   type=int,  help="Number of runs to so", default=2)
    parser.add_argument('-c', '--config', type=str,  help='Path to config file',  default='data/simulated/configs/config.json')
    parser.add_argument('-p', '--params', nargs='*', help='Additional parameters in the form key_value')  # remove?
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)

    # Create subfolder for ensemble
    dir = create_ouput_directory(args.script, config)

    # Update config
    # with specific seed
    # also add new parameters?

    # Prepare the command to run the simulation
    command = [
        'python', 
        args.script,
        '--dir', dir,
        '--config', args.config
    ]
    subprocess.run(command, check=True)
    

    # Timestamp for saving results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save the results
    #print(f"All simulations completed. Results saved in: {output_path}")
