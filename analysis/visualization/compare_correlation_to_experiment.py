import sys
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

sys.path.append("analysis/visualization")
from plot_correlations import plot_correlation, sort_files, initialize_figure

sys.path.append("analysis/utils")
import config_functions as config

sys.path.append("exe/utils")
from vm_functions import hexagon_side

sys.path.append("analysis/experimental")
from data_class import AutocorrelationData, SegmentationData

date    = datetime.now().strftime('%Y%m%d')
fig_dir = f"results/{date}/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)


# Define paths
config_dir = "data/simulated/configs/"

parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
parser.add_argument('param',                 type=str, help="Parameter to plot correlation of (varvar)")
parser.add_argument('var',                   type=str, help="Correlation variable (t or r)")
parser.add_argument('-f', '--filepattern',   type=str, help="Path to files to plot. Typically 'data/simulated/obj/file'. Filename is on form <'path/to/file'>*.autocorr", default=None)
parser.add_argument('-l','--legend',         type=str, help="Add legend (str)",                         default='')
parser.add_argument('-c','--cmap',           type=str, help="Specify matplotlib colormap (str)",        default='plasma')
parser.add_argument('-o','--outdir',         type=str, help="Output directory",                         default="results/")
parser.add_argument('-x', '--xlim',          type=float, help="Upper limit on x-axis", default=9999)
parser.add_argument('-y', '--ylim',          type=float, help="Upper limit on x-axis", default=1.1)
parser.add_argument('-u', '--units',         type=str, help="Unit to plot in (sim or exp)", default='exp')
parser.add_argument('--fmt',    default="-")
parser.add_argument('--binsize',    type=int,   help="size of density bin",    default=200)
parser.add_argument('--figscale',   type=float, help="scaling of figure size", default=1)
parser.add_argument('--frame_to_h', type=float, help="convert frame to h",     default=1/12)
parser.add_argument('-r', '--return_plot', action="store_true")
parser.add_argument('--xlog', action="store_true")
parser.add_argument('--ylog', action="store_true")
parser.add_argument('--log',  action="store_true")
args = parser.parse_args()


# load experimental data
density = 1900

# Add experimental data
dataset = "holomonitor_20240301_B1-4"
#dataset = "holomonitor_20240319_A1-13"

cellprop = SegmentationData()
cellprop.load(f"data/experimental/processed/{dataset}/cell_props.p")
cellcorr  = AutocorrelationData(f"data/experimental/processed/{dataset}/cell_autocorr.p")
fieldcorr = AutocorrelationData(f"data/experimental/processed/{dataset}/field_autocorr.p")
cellprop.add_density()
print(cellcorr.log)
print(cellcorr.temporal_cell)
mask = (cellprop.density > density - args.binsize / 2) * (cellprop.density < density + args.binsize / 2)

if args.param == "vv":
    mask = mask[:-1]


# take mean of correlations at this density
if args.var == "r":
    mean_correlation = np.ma.mean(cellcorr.spatial[args.param][mask], axis=0)
    std_correlation  = np.ma.std(cellcorr.spatial[args.param][mask], axis=0)

elif args.var == "t":
    mean_correlation = np.ma.mean(cellcorr.temporal[args.param][mask], axis=0)
    std_correlation  = np.ma.std(cellcorr.temporal[args.param][mask], axis=0)

else:
    mean_correlation = np.ma.mean(cellcorr.temporal_cell[args.param][mask], axis=0)
    std_correlation  = np.ma.std( cellcorr.temporal_cell[args.param][mask], axis=0)

# plot simulation
fig = initialize_figure(args.param, args, args.figscale)
labels = ['exp']

if args.filepattern != None:
    # Sort data sets by legend value
    files_list, labels_list = sort_files(args.filepattern, args.legend)

    # Define line colors
    cmap   = mpl.colormaps[args.cmap]
    colors = cmap(np.linspace(0.1, 0.9, len(files_list)))

    # Plot each data set
    i = 0
    
    for file, label, color in zip(files_list, labels_list, colors):

        # # config
        # fname = Path(file).stem
        # config_path = f"{config_dir}/{fname}.json"
        # config_file = config.load(config_path)

        plot_correlation(file, label, color, args)
        labels.append(i)
        i += 1

if args.var == "r":
    plt.plot(cellcorr.r_array[args.param], mean_correlation, "b--", label="Exp")
    # plt.fill_between(cellcorr.r_array[args.param], 
    #                  mean_correlation - 1.96*std_correlation, 
    #                  mean_correlation + 1.96*std_correlation,
    #                  alpha=0.5, color="b")
    out_path = f"{fig_dir}spatial_autocorrelation_{args.param}.png"

else:
    plt.plot(cellcorr.t_array[args.param] * args.frame_to_h, mean_correlation, "b--", label="Exp")
    # plt.fill_between(cellcorr.t_array[args.param] * args.frame_to_h, 
    #                  mean_correlation - 1.96*std_correlation, 
    #                  mean_correlation + 1.96*std_correlation,
    #                  alpha=0.5, color="b")
    out_path = f"{fig_dir}temporal_autocorrelation_{args.param}.png"




if args.legend != '':
    fig.legend(title=rf'${args.legend}$')
else:
    fig.legend(labels)
fig.tight_layout()
fig.savefig(out_path)