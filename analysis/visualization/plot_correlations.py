from cells.bind import VertexModel

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# append the path of the parent directory
sys.path.append("analysis/")

#import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import utils.vm_output_handling as vm_output
from utils.correlation_object import VMAutocorrelationObject


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
        plt.hlines(0, 0, 50, linestyle="dashed", color="gray")

    plt.tight_layout()



def plot_results(corr_obj, variable_name):
    """Plot the simulation results."""
    #sns.set(style="whitegrid")
    
    # Assuming the data has columns 'time' and 'value'
    #sns.lineplot(x='time', y='value', data=data, marker='o', label='Simulation Value')
    plt.plot(corr_obj.r_array[variable_name], corr_obj.spatial[variable_name])



def save_plot(figure, output_path):
    """Save the generated plot to a specific directory."""
    figure.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot correlations")
    parser.add_argument('fpattern', type=str, help="filepattern to plot")
    parser.add_argument('variable', type=str, help="variable to plot correlation of")
    parser.add_argument('type',     type=str, help="type of correlation (t or r)")
    parser.add_argument('-cmap',    type=str, help="Matplob colormap", default='plasma')
    args = parser.parse_args()

    assert args.type in ['r', 't']
    
    initialize_figure(args.variable, args.type)

    nfiles = (len(list(Path("data/simulated/obj/").glob(f"{args.fpattern}*.autocorr"))))
    cmap   = mpl.colormaps[args.cmap]
    colors = cmap(np.linspace(0.1, 0.9, nfiles))

    for color, path in zip(colors, Path("data/simulated/obj/").glob(f"{args.fpattern}*.autocorr")):

        # File paths
        fname = f"{args.fpattern}{os.path.basename(path).split(args.fpattern)[-1]}"

        # Load data
        corr_obj = VMAutocorrelationObject(fname)
        
        # Generate plot
        if args.type == "r":
            plt.plot(corr_obj.r_array[args.variable], corr_obj.spatial[args.variable], color=color)
            output_path = f"results/spatial_autocorrelation_{args.variable}_{args.fpattern}.png"
        
        else:
            plt.plot(corr_obj.t_array[args.variable], corr_obj.temporal[args.variable], color=color)
            output_path = f"results/temporal_autocorrelation_{args.variable}_{args.fpattern}.png"
        
    # Save plot
    save_plot(plt, output_path)


if __name__ == "__main__":
    main()

