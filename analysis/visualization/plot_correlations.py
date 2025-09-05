from cells.bind import VertexModel

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Append the path of parent directories
sys.path.append("analysis/")
sys.path.append("VertexModel/analysis/")

# Avoid localhost error on my machine
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import utils.config_functions as     config
from utils.path_handling      import decompose_input_path
from utils.correlation_object import VMAutocorrelationObject

# Autocorrelation output directory
autocorr_dir = "data/simulated/obj/"
config_dir   = "data/simulated/configs/"

# params
rmax = 20
tmax = 99


def sort_files(fnames, legend, parent=''):
    """ 
    Goes through files to plot and returns sorted arrays of legend labels and files

    Parameters:
    - fnames: file name pattern. File names on form <fnames>.autocorr
    - legend: key in config that is used to label plot. Also used as title on legend.
    - parent: path from autocorr_dir to file
    """

    file_list  = []
    label_list = []

    # Aquire labels from config
    for path in Path(f"{autocorr_dir}{parent}").glob(f"{fnames}.autocorr"):

        # File path
        fname = f"{Path(path).stem}"

        # Load config to get plot label
        config_path = f"{config_dir}{fname}.json"
        config_file = config.load(config_path)

        # save in arrays
        file_list.append(fname)
        label_list.append(config.get_value(config_file, legend))

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
        plt.xlabel(r'$r~/~r_0$')
        plt.hlines(0, 0, rmax, linestyle="dashed", color="gray")

    else:
        plt.title(rf'$C_{{{varname}}}(t)$')
        plt.xlabel(r'$t$')
        plt.hlines(0, 0, tmax, linestyle="dashed", color="gray")



def save_plot(figure, out_path):
    """ Save the generated plot to a specific directory. """
    figure.savefig(out_path)
    print(f"Plot saved to {out_path}")



def main():
    parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
    parser.add_argument('filepath', type=str, help="Defines path to files to plot. Filename is on form <'path/to/file*'>.autocorr")
    parser.add_argument('variable', type=str, help="Variable to plot correlation of (varvar)")
    parser.add_argument('type',     type=str, help="Type of correlation (t or r)")
    parser.add_argument('-legend',  type=str, help="Add legend (str)",                  default='')
    parser.add_argument('-cmap',    type=str, help="Specify matplotlib colormap (str)", default='plasma')
    args = parser.parse_args()

    # Decompose input path
    parent, filename = decompose_input_path(args.filepath, autocorr_dir)

    # Assert temporal or spatial correlation
    assert args.type in ['r', 't'], "Wrong correlation variable. Must be r or t"

    # Create figure and plot line at 0 
    initialize_figure(args.variable, args.type)

    # Sort data sets by legend value
    files, labels = sort_files(filename, args.legend, parent)

    # Assert correct file name
    assert len(files) > 0, f"No files matches filename: {args.filepath}.autocorr"

    # Define line colors
    cmap   = mpl.colormaps[args.cmap]
    colors = cmap(np.linspace(0.1, 0.9, len(files)))


    # Plot each data set
    for fname, label, color in zip(files, labels, colors):

        # Load data
        corr_obj = VMAutocorrelationObject(fname)
        
        # Plot
        if args.type == "r":
            out_path = f"results/spatial_autocorrelation_{args.variable}_{filename}.png"

            plt.plot(corr_obj.r_array[args.variable], corr_obj.spatial[args.variable],
                     color=color,
                     label=label)
        
        else:
            out_path = f"results/temporal_autocorrelation_{args.variable}_{filename}.png"

            plt.plot(corr_obj.t_array[args.variable], corr_obj.temporal[args.variable], 
                     color=color, 
                     label=label)
        
    # Add legend
    if args.legend != '':
        plt.legend(title=rf'$\{args.legend}$')

    # Save plot
    plt.tight_layout()
    save_plot(plt, out_path)


if __name__ == "__main__":
    main()

