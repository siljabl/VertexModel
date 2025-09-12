from cells.bind import VertexModel

import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path

# Append the path of relative_parent directories
sys.path.append("analysis/")
sys.path.append("VertexModel/analysis/")

# Avoid localhost error on my machine
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import utils.config_functions as     config
from utils.path_handling      import decompose_input_path
from utils.correlation_object import VMAutocorrelationObject

# Define paths
obj_dir    = "data/simulated/obj/"
fig_dir    = "results/"
config_dir = "data/simulated/configs/"

parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
parser.add_argument('filepath', type=str, help="Path to files to plot. Typically 'data/simulated/obj/file'. Filename is on form <'path/to/file'>*.autocorr")
parser.add_argument('param',    type=str, help="Parameter to plot correlation of (varvar)")
parser.add_argument('var',      type=str, help="Correlation variable (t or r)")
parser.add_argument('--legend', type=str, help="Add legend (str)",                  default='')
parser.add_argument('--cmap',   type=str, help="Specify matplotlib colormap (str)", default='plasma')
parser.add_argument('--shared_config',    help="Shared config files",               action='store_true')

args = parser.parse_args()

# Allow variety of inputs
relative_path = args.filepath.split(obj_dir)[-1]

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
    obj_dir = f"data/simulated/obj/{dir}"
    fig_dir = f"results/{dir}"

    # update config path
    if args.shared_config:
        config_path = f"{config_dir}{dir.split('/')[0]}.json"
    else:
        config_dir = f"data/simulated/configs/{dir}"
    
print(f"\nFiles in: {obj_dir}\nConfig in {config_dir}.\n")

# Subdirectory exists, and create if not
Path(f"{fig_dir}").mkdir(parents=True, exist_ok=True)



# Aquire labels from config
file_list  = []
label_list = []
for path in glob.glob(f"{obj_dir}*.autocorr"):

    # File path
    fname = f"{Path(path).stem}"

    # Load config to get plot label
    config_path = f"{config_dir}{fname}.json"
    config_file = config.load(config_path)

    # Get values from
    label = config.get_value(config_file, args.legend)

    # save in arrays
    file_list.append(f"{Path(path).stem}")
    label_list.append(label)

# Sort labels if legend is specified
if args.legend != '':
    
    # Sort according to label value
    sorted_inds = np.argsort(label_list)
    file_list  = np.array(file_list)[sorted_inds]
    label_list = np.array(label_list)[sorted_inds]


assert args.var in ['r', 't'], "Wrong correlation variable. Must be r or t"
assert len(file_list) > 0, f"No files matches filename: {args.filepath}*.autocorr"




# Create figure
plt.figure(figsize=(6, 4), dpi=300)

if args.var == 'r':
    plt.title(rf'$C_{{{args.param}}}(r)$')
    plt.xlabel(r'$r~/~r_6^*$')
    plt.axhline(0, 0, 1, linestyle="dashed", color="gray")

else:
    plt.title(rf'$C_{{{args.param}}}(t)$')
    plt.xlabel(r'$t~/~\tau_p$')
    plt.axhline(0, 0, 1, linestyle="dashed", color="gray")



# Define line colors
cmap   = mpl.colormaps[args.cmap]
colors = cmap(np.linspace(0.1, 0.9, len(file_list)))


# Plot each data set
for fname, label, color in zip(file_list, label_list, colors):

    # Load data
    corr_obj = VMAutocorrelationObject(out_path=f"{obj_dir}{fname}")

    # Load config
    config_path = f"{config_dir}{fname}.json"
    config_file = config.load(config_path)

    # Plot
    if args.var == "r":
        plt.plot(corr_obj.r_array[args.param], corr_obj.spatial[args.param],
                    '-',
                    color=color,
                    label=label)
    
    else:
        # Get persistence time 
        taup = config.get_value(config_file, 'taup')
        plt.plot(corr_obj.t_array[args.param] / taup, corr_obj.temporal[args.param], 
                    '-',
                    color=color, 
                    label=label)
    
# Add legend
if args.legend != '':
    plt.legend(title=rf'${args.legend}$')

if args.var == "r": 
    out_path = f"{fig_dir}spatial_autocorrelation_{args.param}.png"
else:
    out_path = f"{fig_dir}temporal_autocorrelation_{args.param}.png"

plt.tight_layout()
plt.savefig(out_path)
print(f"Plot saved to {out_path}")


# if __name__ == "__main__":
#     main()

