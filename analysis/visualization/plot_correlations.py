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
obj_dir    = "data/simulated/obj/average/"
fig_dir    = "results/"
config_dir = "data/simulated/configs/"


def sort_files(fnames, legend, relative_parent=''):
    """ 
    Goes through files to plot and returns sorted arrays of legend labels and files

    Parameters:
    - fnames: file name pattern. File names on form <fnames>*.autocorr
    - legend: key in config that is used to label plot. Also used as title on legend.
    - relative_parent: path from obj_dir to file
    """

    file_list  = []
    label_list = []

    # Load config to get plot label
    config_path = f"data/simulated/configs/{fname}.json"
    config_file = config.load(config_path)

    # Aquire labels from config
    for path in Path(f"{obj_dir}{relative_parent}").glob(f"{fnames}*.autocorr"):

        # File path
        fname = f"{Path(path).stem}"

        # Get values from
        label = config.get_value(config_file, legend)

        # save in arrays
        file_list.append(f"{relative_parent}{fname}")
        label_list.append(label)

    # Sort labels if legend is specified
    if legend != '':
        
        # Sort according to label value
        sorted_inds = np.argsort(label_list)
        sorted_file_list  = np.array(file_list)[sorted_inds]
        sorted_label_list = np.array(label_list)[sorted_inds]

        return sorted_file_list, sorted_label_list
    
    else:
        return file_list, label_list



def initialize_figure(varname, type):
    """ Create figure """

    plt.figure(figsize=(6, 4), dpi=300)

    if type == 'r':
        plt.title(rf'$C_{{{varname}}}(r)$')
        plt.xlabel(r'$r~/~r_6^*$')
        plt.axhline(0, 0, 1, linestyle="dashed", color="gray")

    else:
        plt.title(rf'$C_{{{varname}}}(t)$')
        plt.xlabel(r'$t~/~\tau_p$')
        plt.axhline(0, 0, 1, linestyle="dashed", color="gray")



def save_plot(figure, out_path):
    """ Save the generated plot to a specific directory. """
    figure.savefig(out_path)
    print(f"Plot saved to {out_path}")



def main():
    parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
    parser.add_argument('filepath', type=str, help="Path to files to plot. Typically 'data/simulated/obj/file'. Filename is on form <'path/to/file'>*.autocorr")
    parser.add_argument('param',    type=str, help="Parameter to plot correlation of (varvar)")
    parser.add_argument('var',      type=str, help="Correlation variable (t or r)")
    parser.add_argument('--legend', type=str, help="Add legend (str)",                  default='')
    parser.add_argument('--cmap',   type=str, help="Specify matplotlib colormap (str)", default='plasma')
    args = parser.parse_args()

    # Decompose input path
    fname = Path(args.filepath).stem
    dir   = Path(args.filepath).parent

    print(fname, dir)

    # Subdirectory exists, and create if not
    Path(f"{fig_dir}{fname}").mkdir(parents=True, exist_ok=True)

    # Assert temporal or spatial correlation
    assert args.var in ['r', 't'], "Wrong correlation variable. Must be r or t"

    # Create figure and plot line at 0 
    initialize_figure(args.param, args.var)

    # # Sort data sets by legend value
    # files, labels = sort_files(filename, args.legend, relative_parent)

    # Assert correct file name
    #assert len(files) > 0, f"No files matches filename: {args.filepath}*.autocorr"

    # Define line colors
    #cmap   = mpl.colormaps[args.cmap]
    #colors = cmap(np.linspace(0.1, 0.9, len(files)))



    # Load data
    corr_obj = VMAutocorrelationObject(out_path=args.filepath)

    # Load config
    config_path = f"{config_dir}/{fname}.json"
    config_file = config.load(config_path)

    
    # Plot
    if args.var == "r":
        out_path = f"{fig_dir}{fname}/spatial_autocorrelation_{args.param}.png"

        plt.plot(corr_obj.r_array[args.param], corr_obj.spatial[args.param])
    
    else:
        out_path = f"{fig_dir}{fname}/temporal_autocorrelation_{args.param}.png"

        # Get persistence time 
        taup = config.get_value(config_file, 'taup')
        plt.plot(corr_obj.t_array[args.param] / taup, corr_obj.temporal[args.param])
        

    # Save plot
    plt.tight_layout()
    save_plot(plt, out_path)


if __name__ == "__main__":
    main()

