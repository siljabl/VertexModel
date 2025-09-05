from cells.bind import VertexModel

import os
import argparse
import numpy as np
from pathlib import Path

import utils.vm_output_handling as     vm_output
from   utils.correlation_object import VMAutocorrelationObject

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Computes correlations on simulation data and save as pickle")
parser.add_argument('files',       type=str,   help="Defines files to do computations on. Should be full file name (using * if necessary).")
parser.add_argument('--dr',        type=float, help="Spatial step size (float)",                                     default='1')
parser.add_argument('--rmax',      type=float, help="Max distance to coompute correlation for (float)",              default='20')
parser.add_argument('--tmax',      type=float, help="Fraction of total duration to compute correlation for (float)", default='1')
parser.add_argument('--overwrite', type=bool,  help="Overwrite previous computations (True/False)",                  default=False)
args = parser.parse_args()

tmax = 99   # take lenght of array / save in config?

i = 1
for path in Path("data/simulated/raw/").glob(args.files):

    # Load frames as vm objects
    list_vm, init_vm = vm_output.load(path)

    # Get compression/extension
    rho = np.round(42 / init_vm.systemSize[0], 1)

    # Get cell properties
    positions  = vm_output.get_cell_positions(list_vm)
    heights    = vm_output.get_cell_heights(list_vm)
    volumes    = vm_output.get_cell_volumes(list_vm)
    velocities = vm_output.get_cell_velocities(list_vm)

    areas = np.ma.array(volumes / heights)

    # Subtract mean
    h_variation = np.ma.array(heights - np.mean(heights, axis=1, keepdims=True), mask=False)
    A_variation = np.ma.array(areas   - np.mean(areas,   axis=1, keepdims=True), mask=False)
    V_variation = np.ma.array(volumes - np.mean(volumes, axis=1, keepdims=True), mask=False)
    velocities  = np.ma.array(velocities, mask=False)

    # Initialize correlation object
    fname        = os.path.basename(path)
    autocorr_obj = VMAutocorrelationObject(fname)


    # Compute spatial autocorrelations
    autocorr_obj.compute_spatial(positions, h_variation, 'hh', args.dr, args.rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, A_variation, 'AA', args.dr, args.rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, V_variation, 'VV', args.dr, args.rmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, [velocities[:,:,0], velocities[:,:,1]],  'vv', args.dr, args.rmax, t_avrg=True, overwrite=args.overwrite)

    # Compute temporal autocorrelations
    autocorr_obj.compute_temporal(h_variation, 'hh', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(A_variation, 'AA', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(V_variation, 'VV', tmax, t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal([velocities[:,:,0], velocities[:,:,1]],  'vv', tmax, t_avrg=True, overwrite=args.overwrite)

    # Save autocorrelation as .autocorr
    autocorr_obj.save_pickle()