from cells.bind import VertexModel

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# append the path of the parent directory
sys.path.append("analysis/")
sys.path.append("VertexModel/analysis/")

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

from utils.config_functions   import load_config, get_config_value
from utils.correlation_object import VMAutocorrelationObject


def sort_files(fpattern, legend):
    """ 
    Goes through files to plot and returns sorted arrays of legend and file

    Parameters:
    - fpattern: pattern in file name
    - legend: key in config that is used to label plot. Also used as title on legend.
    """

    files  = []
    labels = []

    # Aquire labels from config
    for path in Path("data/simulated/obj/").glob(f"{fpattern}*.autocorr"):

        # File path
        fname = f"{fpattern}{os.path.basename(path).split(fpattern)[-1]}"

        # Load data
        corr_obj = VMAutocorrelationObject(fname)

        # Load config to get plot label
        config_path = f"data/simulated/configs/{corr_obj.fname}.json"
        config = load_config(config_path)

        # save in arrays
        files.append(fname)
        labels.append(get_config_value(config, legend))

    # Sort labels if legend is specified
    if legend != '':
        
        sort_inds = np.argsort(labels)
        sort_files  = np.array(files)[sort_inds]
        sort_labels = np.array(labels)[sort_inds]

        return sort_files, sort_labels
    
    else:
        return files, labels



def initialize_figure(variable_name, type):
    """ Create figure """
    plt.figure(figsize=(6, 4))

    if type == 'r':
        plt.title(rf'$C_{{{variable_name}}}(r)$')
        plt.xlabel(r'$r~/~r_0$')
        plt.hlines(0, 0, 20, linestyle="dashed", color="gray")

    else:
        plt.title(rf'$C_{{{variable_name}}}(t)$')
        plt.xlabel(r'$t$')
        plt.hlines(0, 0, 99, linestyle="dashed", color="gray")



def save_plot(figure, output_path):
    """ Save the generated plot to a specific directory. """
    figure.savefig(output_path)
    print(f"Plot saved to {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Plot correlations")
    parser.add_argument('fpattern', type=str, help="filepattern to plot")
    parser.add_argument('variable', type=str, help="variable to plot correlation of")
    parser.add_argument('type',     type=str, help="type of correlation (t or r)")
    parser.add_argument('-legend',  type=str, help="title of legend", default='')
    parser.add_argument('-cmap',    type=str, help="Matplob colormap", default='plasma')
    args = parser.parse_args()

    assert args.type in ['r', 't']

    # create figure and plot line at 0 
    initialize_figure(args.variable, args.type)

    # sort data sets by legend value
    files, labels = sort_files(args.fpattern, args.legend)

    # define line colors
    cmap   = mpl.colormaps[args.cmap]
    colors = cmap(np.linspace(0.1, 0.9, len(files)))


    # plot from each data set
    for fname, label, color in zip(files, labels, colors):

        # Load data
        corr_obj = VMAutocorrelationObject(fname)
        
        # Generate plot
        if args.type == "r":
            output_path = f"results/spatial_autocorrelation_{args.variable}_{args.fpattern}.png"

            plt.plot(corr_obj.r_array[args.variable], corr_obj.spatial[args.variable],
                     color=color,
                     label=label)
        
        else:
            output_path = f"results/temporal_autocorrelation_{args.variable}_{args.fpattern}.png"

            plt.plot(corr_obj.t_array[args.variable], corr_obj.temporal[args.variable], 
                     color=color, 
                     label=label)
        
    # Add legend
    if args.legend != '':
        plt.legend(title=rf'$\{args.legend}$')

    # Save plot
    plt.tight_layout()
    save_plot(plt, output_path)


if __name__ == "__main__":
    main()

