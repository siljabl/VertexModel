from cells.bind import VertexModel

import os
import sys
import glob
import platform
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
config_dir = "data/simulated/configs/"


if platform.node() != 'silja-work':
    config_dir = "../../../../hdd_data/silja/VertexModel_data/simulated/configs/"                                                                                                                                                                     


def sort_files(fpattern, legend):
    """ 
    Goes through files to plot and returns sorted arrays of legend labels and files

    Parameters:
    - fnames: file name pattern. File names on form <fnames>*.autocorr
    - legend: key in config that is used to label plot. Also used as title on legend.
    """

    file_list  = []
    label_list = []

    # Aquire labels from config
    for path in glob.glob(f"{fpattern}*.autocorr"):

        fname = Path(path).stem
        if 'seed' in fname:
            fname = Path(path).parent.stem

        # Load config to get plot label
        config_path = f"{config_dir}/{fname}.json"
        config_file = config.load(config_path)

        # Get values from
        label = config.get_value(config_file, legend)

        # save in arrays
        file_list.append(path)
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

    fig = plt.figure(figsize=(6, 4), dpi=300)

    if type == 'r':
        plt.title(rf'$C_{{{varname}}}(r)$')
        plt.xlabel(r'$r~/~r_6^*$')
        plt.axhline(0, 0, 1, linestyle="dashed", color="gray")

    else:
        plt.title(rf'$C_{{{varname}}}(t)$')
        plt.xlabel(r'$t$')
        #plt.xlabel(r'$t~/~\tau_p$')
        plt.axhline(0, 0, 1, linestyle="dashed", color="gray")

    return fig



def save_plot(figure, out_path):
    """ Save the generated plot to a specific directory. """
    figure.savefig(out_path)
    print(f"Plot saved to {out_path}")


def plot_correlation(path, label, color, args):
    # Load data
    corr_obj = VMAutocorrelationObject(out_path=path)

    # Load config
    fname = Path(path).stem
    if 'seed' in fname:
        fname = Path(path).parent.stem
        fig_dir = f"results/{fname}/"
        Path(f"{fig_dir}").mkdir(parents=True, exist_ok=True)

    config_path = f"{config_dir}{fname}.json"
    config_file = config.load(config_path)

    # Plot
    if args.var == "r":
        x = corr_obj.r_array[args.param]
        y = corr_obj.spatial[args.param]
        plt.plot(x[(x <= args.xlim) * (y <= args.ylim)], y[(x <= args.xlim) * (y <= args.ylim)],
                    args.fmt,
                    color=color,
                    label=label)
    
    else:
        # Get persistence time
        taup = config.get_value(config_file, 'taup')
        x = corr_obj.t_array[args.param]# / taup
        y = corr_obj.temporal[args.param]
        plt.plot(x[(x <= args.xlim) * (y <= args.ylim)], y[(x <= args.xlim) * (y <= args.ylim)],
                    args.fmt,
                    color=color, 
                    label=label)




def main():
    fig_dir = "results/"

    parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
    parser.add_argument('filepattern',   type=str, help="Path to files to plot. Typically 'data/simulated/obj/file'. Filename is on form <'path/to/file'>*.autocorr")
    parser.add_argument('param',         type=str, help="Parameter to plot correlation of (varvar)")
    parser.add_argument('var',           type=str, help="Correlation variable (t or r)")
    parser.add_argument('-l','--legend', type=str, help="Add legend (str)",                         default='')
    parser.add_argument('-c','--cmap',   type=str, help="Specify matplotlib colormap (str)",        default='plasma')
    parser.add_argument('-o','--outdir', type=str, help="Output directory",                         default="results/")
    parser.add_argument('-x', '--xlim',  type=float, help="Upper limit on x-axis", default=9999)
    parser.add_argument('-y', '--ylim',  type=float, help="Upper limit on x-axis", default=1.1)
    parser.add_argument('--xlog', action="store_true")
    parser.add_argument('--ylog', action="store_true")
    parser.add_argument('--log',  action="store_true")
    parser.add_argument('--fmt',    default="-")
    parser.add_argument('-r', '--return_plot', action="store_true")
    args = parser.parse_args()

    # Assert temporal or spatial correlation
    assert args.var in ['r', 't'], "Wrong correlation variable. Must be r or t"


    # Sort data sets by legend value
    files_list, labels_list = sort_files(args.filepattern, args.legend)

    assert len(files_list) > 0, f"No files matches filename: {args.filepattern}*.autocorr"


    # Define line colors
    cmap   = mpl.colormaps[args.cmap]
    colors = cmap(np.linspace(0.1, 0.9, len(files_list)))


    # Plot each data set
    fig = initialize_figure(args.param, args.var)
    for file, label, color in zip(files_list, labels_list, colors):

        plot_correlation(file, label, color, args)
        
    # Add legend
    if args.legend != '':
        fig.legend(title=rf'${args.legend}$')

    if args.var == "r":
        out_path = f"{fig_dir}spatial_autocorrelation_{args.param}.png"
    else:
        out_path = f"{fig_dir}temporal_autocorrelation_{args.param}.png"

    # log-scale
    if args.xlog or args.log:
        plt.xscale("log")
    if args.ylog or args.log:
        plt.yscale("log")

    if args.return_plot:
        return fig

    else:
        fig.tight_layout()
        fig.savefig(out_path)
        print(f"Plot saved to {out_path}")



if __name__ == "__main__":
    main()

