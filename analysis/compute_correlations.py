from cells.bind import VertexModel

import glob
import argparse
import numpy as np
from pathlib import Path

import utils.config_functions   as config
import utils.vm_output_handling as vm_output

from utils.correlation_object import VMAutocorrelationObject

# Define paths
obj_dir    = "data/simulated/obj/"
data_dir   = "data/simulated/raw/"
config_dir = "data/simulated/configs/"

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Computes correlations on simulation data and save as pickle")
parser.add_argument('filepath',    type=str,   help="Defines path to file or dir, typically: data/simulated/raw/dir/.")
parser.add_argument('--dr',        type=float, help="Spatial step size (float)",                                            default='1')
parser.add_argument('--rfrac',     type=float, help="Max distance to compute correlation for (float)",                      default='0.5')
parser.add_argument('--tfrac',     type=float, help="Fraction of total duration to compute correlation for (float)",        default='0.5')
parser.add_argument('--mean_var',  type=str,   help="Variable to take mean over in <x - <x>_var> (t or cell). Default: t",  default='t')
parser.add_argument('--overwrite',             help="Overwrite previous computations",   action='store_true')
args = parser.parse_args()


# Allow variety of inputs
relative_path = args.filepath.split(data_dir)[-1]

# file is in subdir
if len(relative_path.split("/")) > 1:

    # input is directory
    if relative_path.split("/")[-1] == "":
        dir = relative_path

    # file is in subdirectory (only works if fname is not in dirname)
    else:
        fname = relative_path.split("/")[-1]
        dir = relative_path.split(fname)[0]

    # update paths for input and output
    obj_dir     = f"data/simulated/obj/{dir}"
    data_dir    = f"data/simulated/raw/{dir}"
    config_path = f"{config_dir}{dir.split('/')[0]}.json"

    # only states of ensembles are saved in common folder
    ensemble = True

else:
    ensemble = False
    
print(f"Computing correlations of files in {data_dir} with config files in {config_dir}.\n")


# Subdirectory exists, and create if not
Path(f"{obj_dir}").mkdir(parents=True, exist_ok=True)



for path in glob.glob(f"{args.filepath}*"):

    # Load frames as vm objects
    list_vm, init_vm = vm_output.load(path)

    # run specific config file
    if not ensemble:
        config_path = f"{config_dir}{Path(path).stem}.json"

    # Load config
    config_file = config.load(config_path)

    # Get values from config
    rhex    = config.get_value(config_file, 'rhex') 
    Nframes = config.get_value(config_file, 'Nframes')
    Lgrid   = config.get_value(config_file, 'Lgrid')

    # Get cell properties
    positions  = vm_output.get_cell_positions(list_vm)
    heights    = vm_output.get_cell_heights(list_vm)
    volumes    = vm_output.get_cell_volumes(list_vm)
    velocities = vm_output.get_cell_velocities(list_vm)

    areas = np.ma.array(volumes / heights)

    # Define mean variable axis
    if args.mean_var == 't':
        mean_var = 1
    elif args.mean_var == 'cell': 
        mean_var = 0

    # Subtract mean
    h_variation = np.ma.array(heights - np.mean(heights, axis=mean_var, keepdims=True), mask=False)
    A_variation = np.ma.array(areas   - np.mean(areas,   axis=mean_var, keepdims=True), mask=False)
    V_variation = np.ma.array(volumes - np.mean(volumes, axis=mean_var, keepdims=True), mask=False)
    velocities  = np.ma.array(velocities, mask=False)
    velocities  = [velocities[:,:,0], velocities[:,:,1]]

    # Initialize correlation object
    autocorr_obj = VMAutocorrelationObject(in_path=path)

    # Upper limit on distance
    rmax = Lgrid * args.rfrac

    # Compute spatial autocorrelations
    autocorr_obj.compute_spatial(positions, h_variation, 'hh', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, A_variation, 'AA', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, V_variation, 'VV', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, velocities,  'vv', args.dr, rmax, t_avrg=True, overwrite=args.overwrite) 

    # Upper limit on t ime difference
    tmax = int(Nframes * args.tfrac)

    # Compute temporal autocorrelations
    autocorr_obj.compute_temporal(h_variation, 'hh', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(A_variation, 'AA', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(V_variation, 'VV', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(velocities,  'vv', tmax, t_avrg=True, overwrite=args.overwrite)

    # Save autocorrelation as .autocorr
    autocorr_obj.save_pickle()