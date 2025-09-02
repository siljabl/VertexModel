from cells.bind import VertexModel

import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from operator import itemgetter
from utils.file_functions import load_vm_output
from utils.correlations import general_spatial_correlation, general_temporal_correlation


dr = 2


#fig, ax = plt.subplots(1,3, figsize=(12,3))

for file in Path("../../data/simulated/raw/").glob("nodivision_20250902_1328.p"):
    print(file)

    # load frames
    list_vm, init_vm = load_vm_output(file)                       # stop when we have read the whole file

        
    # # get cell properties
    # cells = list_vm[0].getVertexIndicesByType("centre")                 # indices of cell centres (from first frame)

    # positions = np.ma.array(list(map(
    #     lambda vm: itemgetter(*cells)(vm.getPositions(wrapped=False)),  # unwrapped positions of centres
    #     list_vm)))

    # heights = np.ma.array(list(map(
    # lambda vm: itemgetter(*cells)(vm.vertexForces["surface"].height.copy()),
    # list_vm)))

    # volumes = np.ma.array(list(map(
    #     lambda vm: itemgetter(*cells)(vm.vertexForces["surface"].volume.copy()),  # unwrapped positions of centres
    #     list_vm)))

    # areas = np.ma.array(volumes / heights)

    # # subtract mean
    # h_var = heights - np.mean(heights, axis=1, keepdims=True)
    # A_var = areas   - np.mean(areas,   axis=1, keepdims=True)
    # V_var = volumes - np.mean(volumes, axis=1, keepdims=True)


    # r_max = int(vm.systemSize[0] / 2)
    # C_hh = general_spatial_correlation(positions[:,:,0], positions[:,:,1], A_var, dr=dr, r_max=r_max)


    # plt.plot(C_hh['r_bin_centers'], np.mean(C_hh['C_norm'], axis=0))

