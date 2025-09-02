import os, sys
print(os.path.dirname(sys.executable))

from cells.bind import VertexModel

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from operator import itemgetter
import utils.vm_output_handling as output
from utils.correlations import general_spatial_correlation, general_temporal_correlation

dr    = 2
r_max = 20


fig, ax = plt.subplots(1,1, figsize=(5,3))

for file in Path("data/simulated/raw/").glob("nodivision_20250902_1328.p"):
    print(file)

    # load frames
    list_vm, init_vm = output.load(file)                       # stop when we have read the whole file

    # get cell properties
    positions = output.get_cell_positions(list_vm)
    heights   = output.get_cell_heights(list_vm)
    volumes   = output.get_cell_volumes(list_vm)

    areas = np.ma.array(volumes / heights)


    # subtract mean
    h_var = heights - np.mean(heights, axis=1, keepdims=True)
    A_var = areas   - np.mean(areas,   axis=1, keepdims=True)
    V_var = volumes - np.mean(volumes, axis=1, keepdims=True)

    # compute correlation
    C_hh = general_spatial_correlation(positions[:,:,0], positions[:,:,1], h_var, dr=dr, r_max=r_max, t_avrg=True)
    C_AA = general_spatial_correlation(positions[:,:,0], positions[:,:,1], A_var, dr=dr, r_max=r_max)
    C_VV = general_spatial_correlation(positions[:,:,0], positions[:,:,1], V_var, dr=dr, r_max=r_max)


    ax.plot(C_hh['r_bin_centers'], C_hh['C_norm'])

fig.savefig("test.png")

