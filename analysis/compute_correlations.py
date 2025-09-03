from cells.bind import VertexModel

import os
import numpy as np
from pathlib import Path

import utils.vm_output_handling as vm_output
from utils.correlation_object import VMAutocorrelationObject

# turn off interactive plotting
#mpl.use('Agg')

dr    = 2
r_max = 20


i = 1
for path in Path("data/simulated/raw/").glob("nodivision_20250902_*.p"):
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
    h_variation = heights - np.mean(heights, axis=1, keepdims=True)
    A_variation = areas   - np.mean(areas,   axis=1, keepdims=True)
    V_variation = volumes - np.mean(volumes, axis=1, keepdims=True)

    # initialize correlation object
    autocorr_obj = VMAutocorrelationObject(fname)
    autocorr_obj.compute_spatial(positions, h_variation, dr, r_max, 'hh', t_avrg=True)
    autocorr_obj.compute_spatial(positions, A_variation, dr, r_max, 'AA', t_avrg=True)
    autocorr_obj.compute_spatial(positions, V_variation, dr, r_max, 'VV', t_avrg=True)

    # save autocorrelation
    autocorr_obj.save_pickle()