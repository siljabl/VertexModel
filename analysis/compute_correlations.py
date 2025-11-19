from cells.bind import VertexModel

import glob
import argparse
import platform
import numpy as np
from pathlib import Path
from multiprocessing import Pool

import utils.config_functions   as config
import utils.vm_output_handling as vm_output

from utils.correlation_object import VMAutocorrelationObject


def vm_compute_correlation(path, config_path, args):

    # Load config
    config_file = config.load(config_path)

    # Compute time period between frames
    dt = config_file["simulation"]["dt"]
    T  = config_file["simulation"]["period"]
    df = T * dt

    # Load frames as vm objects
    list_vm, init_vm = vm_output.load(path, df=df)
    print("Lenght of data: ", len(list_vm))


    # Get values from config
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
    velocities  = [np.ma.array(velocities[:,:,0] - np.ma.mean(velocities[:,:,0], axis=mean_var, keepdims=True), mask=False),
                   np.ma.array(velocities[:,:,1] - np.ma.mean(velocities[:,:,1], axis=mean_var, keepdims=True), mask=False)]

    # Initialize correlation object
    autocorr_obj = VMAutocorrelationObject(in_path=path)

    if args.var == 'r' or args.var == 'all':
        # Upper limit on distance
        rmax = Lgrid * args.rfrac

        # Compute spatial autocorrelations
        if args.param == 'hh' or args.param == 'all':
            autocorr_obj.compute_spatial(positions, h_variation, 'hh', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
        if args.param == 'AA' or args.param == 'all':
            autocorr_obj.compute_spatial(positions, A_variation, 'AA', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
        if args.param == 'VV' or args.param == 'all':
            autocorr_obj.compute_spatial(positions, V_variation, 'VV', args.dr, rmax, t_avrg=True, overwrite=args.overwrite)
        if args.param == 'vv' or args.param == 'all':
            autocorr_obj.compute_spatial(positions, velocities,  'vv', args.dr, rmax, t_avrg=True, overwrite=args.overwrite) 

    if args.var == 't' or args.var == 'all':
        # Upper limit on t ime difference
        tmax = int(Nframes * args.tfrac)

        # Compute temporal autocorrelations
        if args.param == 'hh' or args.param == 'all':
            autocorr_obj.compute_temporal(h_variation, 'hh', tmax, df=df, t_avrg=True, overwrite=args.overwrite)
        if args.param == 'AA' or args.param == 'all':
            autocorr_obj.compute_temporal(A_variation, 'AA', tmax, df=df, t_avrg=True, overwrite=args.overwrite)
        if args.param == 'VV' or args.param == 'all':
            autocorr_obj.compute_temporal(V_variation, 'VV', tmax, df=df, t_avrg=True, overwrite=args.overwrite)
        if args.param == 'vv' or args.param == 'all':
            autocorr_obj.compute_temporal(velocities,  'vv', tmax, df=df, t_avrg=True, overwrite=args.overwrite)

    # Save autocorrelation as .autocorr
    autocorr_obj.save_pickle()



def main():

    # Define paths
    obj_dir    = "data/simulated/processed/"
    data_dir   = "data/simulated/raw/"
    config_dir = "data/simulated/configs/"

    if platform.node() != 'silja-work':
        print(f"Running simulation from {platform.node()}")
        obj_dir    = "../../../../hdd_data/silja/VertexModel_data/simulated/processed/"
        data_dir   = "../../../../hdd_data/silja/VertexModel_data/simulated/raw/"
        config_dir = "../../../../hdd_data/silja/VertexModel_data/simulated/configs/"
        print(f"Saving output in {obj_dir}")



    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Computes correlations on simulation data and save as pickle")
    parser.add_argument('filepath',          type=str, help="Defines path to file or dir, typically: data/simulated/raw/dir/.")
    parser.add_argument('-p', '--param',     type=str, help="Parameter to plot correlation of (varvar)", default="all")
    parser.add_argument('-v', '--var',       type=str, help="Correlation variable (t or r)",             default="all")
    parser.add_argument('-o','--overwrite',            help="Overwrite previous computations",           action='store_true')
    parser.add_argument('--dr',            type=float, help="Spatial step size (float)",                                            default='1')
    parser.add_argument('--rfrac',         type=float, help="Max distance to compute correlation for (float)",                      default='0.5')
    parser.add_argument('--tfrac',         type=float, help="Fraction of total duration to compute correlation for (float)",        default='0.5')
    parser.add_argument('--mean_var',      type=str,   help="Variable to take mean over in <x - <x>_var> (t or cell). Default: t",  default='t')
    args = parser.parse_args()


    # Allow variety of inputs
    relative_path = args.filepath.split(data_dir)[-1]

    # file is in subdir
    if len(relative_path.split("/")) > 1:
        print("Data is part of ensemble")

        # input is directory
        if relative_path.split("/")[-1] == "":
            dir = relative_path

        # file is in subdirectory (only works if fname is not in dirname)
        else:
            fname = relative_path.split("/")[-1]
            dir = relative_path.split(fname)[0]

        # update paths for input and output
        obj_dir     = f"{obj_dir}{dir}"
        data_dir    = f"{data_dir}{dir}"
        config_path = f"{config_dir}{dir.split('/')[0]}.json"

        # only states of ensembles are saved in common folder
        ensemble = True

    else:
        print("Data is not part of ensemble")
        ensemble = False
        
    print(f"Computing correlations of files in {data_dir} with config files in {config_dir}.\n")


    # Subdirectory exists, and create if not
    Path(f"{obj_dir}").mkdir(parents=True, exist_ok=True)

    commands = []
    for path in glob.glob(f"{args.filepath}*"):
        # run specific config file
        if not ensemble:
            config_path = f"{config_dir}{Path(path).stem}.json"
            
        command = [
            path,
            config_path,
            args
        ]
        commands.append(command)
    
    Npool = len(glob.glob(f"{args.filepath}*"))
    if Npool > 16: Npool = 16

    with Pool(processes=Npool) as pool:
        pool.starmap(vm_compute_correlation, commands)


if __name__ == "__main__":
    main()

