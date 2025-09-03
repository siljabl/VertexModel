from cells.bind import VertexModel

import os
import sys
  
# append the path of the parent directory
sys.path.append("analysis/")

import argparse
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

import utils.vm_output_handling as vm_output
from utils.correlation_object import VMAutocorrelationObject


# def plot_results(corr_obj, variable_name):
#     """Plot the simulation results."""
#     plt.figure(figsize=(10, 6))
#     #sns.set(style="whitegrid")
    
#     # Assuming the data has columns 'time' and 'value'
#     #sns.lineplot(x='time', y='value', data=data, marker='o', label='Simulation Value')
#     plt.plot(corr_obj.r_array[variable_name], corr_obj.spatial[variable_name])
    
#     plt.title(r'$C_{hh}(r)$')
#     plt.xlabel(r'$r~/~r_0$')
#     plt.tight_layout()


# def save_plot(figure, output_path):
#     """Save the generated plot to a specific directory."""
#     figure.savefig(output_path)
#     print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot correlations")
    parser.add_argument('file',   type=str,  help="filename")
    args = parser.parse_args()

    # File paths
    output_path = '/results/correlation_plot.png'
    fname = os.path.basename(args.file)
    print(fname)

    # Load data
    corr_obj = VMAutocorrelationObject(fname)
    
    # Generate plot
    #plot_result_figure = plot_results(corr_obj)
    
    # Save plot
    #save_plot(plt, output_path)
    
    # Show plot if desired
    #plt.show()

if __name__ == "__main__":
    main()

