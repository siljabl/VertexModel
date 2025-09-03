from cells.bind import VertexModel

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from operator import itemgetter
import utils.vm_output_handling as output
from analysis.utils.correlation_computations import general_spatial_correlation, general_temporal_correlation

# turn off interactive plotting
mpl.use('Agg')

dr    = 2
r_max = 20


fig, ax = plt.subplots(1,3, figsize=(10,3))
for i in range(3):
    ax[i].hlines(0, 0, 18, linestyle='dashed', color="gray")

cmap = plt.get_cmap('plasma', 6)

i = 1
for file in Path("data/simulated/raw/").glob("nodivision_20250902_*.p"):
    print(file)

    # load frames
    list_vm, init_vm = output.load(file)                       # stop when we have read the whole file

    # get compression/extension
    rho = np.round(42 / init_vm.systemSize[0], 1)

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
    C_AA = general_spatial_correlation(positions[:,:,0], positions[:,:,1], A_var, dr=dr, r_max=r_max, t_avrg=True)
    C_VV = general_spatial_correlation(positions[:,:,0], positions[:,:,1], V_var, dr=dr, r_max=r_max, t_avrg=True)


    ax[0].plot(C_hh['r_bin_centers'] / rho, C_hh['C_norm'], color=cmap(i), label=rho)
    ax[1].plot(C_AA['r_bin_centers'] / rho, C_AA['C_norm'], color=cmap(i))
    ax[2].plot(C_VV['r_bin_centers'] / rho, C_VV['C_norm'], color=cmap(i))

    i += 1

ax[0].set(xlabel=r"$r~/~r_0$", title=r"$C_{hh}(r)$")
ax[1].set(xlabel=r"$r~/~r_0$", title=r"$C_{AA}(r)$")
ax[2].set(xlabel=r"$r~/~r_0$", title=r"$C_{VV}(r)$")

ax[0].legend()
fig.tight_layout()
fig.savefig("results/spatial_correlations_20250902.png")

