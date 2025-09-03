from cells.bind import VertexModel

import os
import argparse
import numpy as np
from pathlib import Path

import utils.vm_output_handling as vm_output
from utils.correlation_object import VMAutocorrelationObject

# command-line argument parsing
parser = argparse.ArgumentParser(description="Compute correlations on simulation data and save as pickle")
parser.add_argument('fpattern',    type=str,   help="file patter to do computations on")
parser.add_argument('--dr',        type=float, help="spatial step size [r_6]",         default='1')
parser.add_argument('--rmax',      type=float, help="max distance to consider [r_6]",  default='20')
parser.add_argument('--overwrite', type=bool,  help="overwrite previous computations", default='False')
args = parser.parse_args()

tmax = 99   # take lenght of array / save in config?

i = 1
for path in Path("data/simulated/raw/").glob(args.fpattern):
    fname = os.path.basename(path)

    # load frames
    list_vm, init_vm = vm_output.load(path)

    # get compression/extension
    rho = np.round(42 / init_vm.systemSize[0], 1)

    # get cell properties
    positions = vm_output.get_cell_positions(list_vm)
    heights   = vm_output.get_cell_heights(list_vm)
    volumes   = vm_output.get_cell_volumes(list_vm)

    areas = np.ma.array(volumes / heights)

    # subtract mean
    h_variation = np.ma.array(heights - np.mean(heights, axis=1, keepdims=True), mask=False)
    A_variation = np.ma.array(areas   - np.mean(areas,   axis=1, keepdims=True), mask=False)
    V_variation = np.ma.array(volumes - np.mean(volumes, axis=1, keepdims=True), mask=False)

    # initialize correlation object
    autocorr_obj = VMAutocorrelationObject(fname)

    # compute spatial autocorrelations
    autocorr_obj.compute_spatial(positions, h_variation, args.dr, args.rmax, 'hh', t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, A_variation, args.dr, args.rmax, 'AA', t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_spatial(positions, V_variation, args.dr, args.rmax, 'VV', t_avrg=True, overwrite=args.overwrite)

    # compute temporal autocorrelations
    autocorr_obj.compute_temporal(h_variation, tmax, 'hh', t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(A_variation, tmax, 'AA', t_avrg=True, overwrite=args.overwrite)
    autocorr_obj.compute_temporal(V_variation, tmax, 'VV', t_avrg=True, overwrite=args.overwrite)

    # save autocorrelation
    autocorr_obj.save_pickle()