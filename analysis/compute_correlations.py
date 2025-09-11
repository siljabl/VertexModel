from cells.bind import VertexModel

import glob
import argparse
import numpy as np
from pathlib import Path

import utils.config_functions   as config
import utils.vm_output_handling as vm_output

from utils.path_handling      import decompose_input_path
from utils.correlation_object import VMAutocorrelationObject

# Define paths
obj_dir    = "data/simulated/obj/"
data_dir   = "data/simulated/raw/"
config_dir = "data/simulated/configs/"

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Computes correlations on simulation data and save as pickle")
parser.add_argument('filepath',    type=str,   help="Defines path to files to do computations on, typically data/simulated/raw/dir/filepattern. filepattern should not be same as dir!")
parser.add_argument('--dr',        type=float, help="Spatial step size (float)",                                            default='1')
parser.add_argument('--rfrac',     type=float, help="Max distance to compute correlation for (float)",                      default='0.5')
parser.add_argument('--tfrac',     type=float, help="Fraction of total duration to compute correlation for (float)",        default='0.5')
parser.add_argument('--mean_var',  type=str,   help="Variable to take mean over in <x - <x>_var> (t or cell). Default: t",  default='t')
parser.add_argument('-o', '--overwrite', type=bool,  help="Overwrite previous computations (True/False)",                   default=False)
args = parser.parse_args()

# Allow variety of input styles
relative_path = args.filepath.split(data_dir)[-1]

# file is in data_dir 
if len(relative_path.split("/")) == 1:
    relative_parent = ""

# input is directory
elif relative_path.split("/")[-1] == "":
    relative_parent = relative_path

# file is in subdirectory (only works if fname is not in dirname)
else:
    fname = relative_path.split("/")[-1]
    relative_parent = relative_path.split(fname)[0]


# Subdirectory exists, and create if not
Path(f"{obj_dir}{relative_parent}").mkdir(parents=True, exist_ok=True)


for path in glob.glob(f"{args.filepath}*"):

    # Load frames as vm objects
    list_vm, init_vm = vm_output.load(path)
    print(path)

    # Load config
    config_path = f"{config_dir}{relative_parent}{Path(path).stem}.json"
    config_file = config.load(config_path)
    print(config_path)

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

    # Initialize correlation object
    print(path)
    autocorr_obj = VMAutocorrelationObject(path)

    # Upper limit on distance
    rmax = Lgrid * args.rfrac

    # Compute spatial autocorrelations
    autocorr_obj.compute_spatial(positions, h_variation, 'hh', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, A_variation, 'AA', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, V_variation, 'VV', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, [velocities[:,:,0], velocities[:,:,1]], 'vv', args.dr, rmax, t_avrg=True, overwrite=args.overwrite) 

    # Upper limit on t ime difference
    tmax = int(Nframes * args.tfrac)

    # Compute temporal autocorrelations
    autocorr_obj.compute_temporal(h_variation, 'hh', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(A_variation, 'AA', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(V_variation, 'VV', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal([velocities[:,:,0], velocities[:,:,1]], 'vv', tmax, t_avrg=True, overwrite=args.overwrite)

    # Save autocorrelation as .autocorr
    autocorr_obj.save_pickle()