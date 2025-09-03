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


def plot_results(corr_obj, variable_name):
    """Plot the simulation results."""
    #sns.set(style="whitegrid")
    
    # Assuming the data has columns 'time' and 'value'
    #sns.lineplot(x='time', y='value', data=data, marker='o', label='Simulation Value')
    plt.plot(corr_obj.r_array[variable_name], corr_obj.spatial[variable_name])
    
    plt.title(r'$C_{hh}(r)$')
    plt.xlabel(r'$r~/~r_0$')
    plt.tight_layout()




def save_plot(figure, output_path):
    """Save the generated plot to a specific directory."""
    figure.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot correlations")
    parser.add_argument('fpattern', type=str, help="filepattern to plot")
    parser.add_argument('var',      type=str, help="var to plot correlation of")
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))
    
    for path in Path("data/simulated/obj/").glob(f"*{args.fpattern}*.obj"):

        # File paths
        fname = f"{args.fpattern}{os.path.basename(path).split(args.fpattern)[-1]}"
        output_path = f"results/spatial_autocorrelation_{args.var}_{args.fpattern}.png"

        # Load data
        corr_obj = VMAutocorrelationObject(fname)
        
        # Generate plot
        plot_results(corr_obj, args.var)
        
        # Save plot
        save_plot(plt, output_path)


if __name__ == "__main__":
    main()

