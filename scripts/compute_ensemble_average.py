import glob
import pickle
import argparse
import subprocess
import numpy as np

from pathlib import Path

from utils.correlation_object import VMAutocorrelationObject

processed_dir = "data/simulated/processed/"
ensemble_dir  = f"{processed_dir}averages/"
Path(ensemble_dir).mkdir(parents=True, exist_ok=True)



def compute_correlations(dirpath, args):
    command = [
        'python', 
        'analysis/compute_correlations.py',
        dirpath,
        '--param',     args.param,
        '--var',       args.var,
        '--overwrite',
        '--dr',        str(args.dr),
        '--rfrac',     str(args.rfrac),
        '--tfrac',     str(args.tfrac),
        '--mean_var',  args.mean_var
    ]

    subprocess.run(command, check=True)
    


def main():
    parser = argparse.ArgumentParser(description="Compute ensemble average of autocorrelation")
    parser.add_argument('dirpath', type=str,      help="Path to ensemble directory. Typically 'data/simulated/processed/dir'")
    parser.add_argument('--compute_correlations', help="Run compute_correlations.py before taking average.", action="store_true")

    # compute correlations inputs
    parser.add_argument('-p', '--param',     type=str, help="Parameter to plot correlation of (varvar)", default="all")
    parser.add_argument('-v', '--var',       type=str, help="Correlation variable (t or r)",             default="all")
    parser.add_argument('--dr',            type=float, help="Spatial step size (float)",                                            default='1')
    parser.add_argument('--rfrac',         type=float, help="Max distance to compute correlation for (float)",                      default='0.5')
    parser.add_argument('--tfrac',         type=float, help="Fraction of total duration to compute correlation for (float)",        default='0.5')
    parser.add_argument('--mean_var',      type=str,   help="Variable to take mean over in <x - <x>_var> (t or cell). Default: t",  default='t')
    args = parser.parse_args()

    if args.compute_correlations:
        compute_correlations(args.dirpath, args)
        
        # update dirpath
        args.dirpath = f"data/simulated/processed/{Path(args.dirpath).stem}/"

    # List all files
    files_list   = glob.glob(f"{args.dirpath}*.autocorr")
    autocorr_tmp = VMAutocorrelationObject(out_path=files_list[0])

    # Initialize ensemble correlation object
    out_path = f"{ensemble_dir}{Path(args.dirpath).stem}"
    autocorr_obj = VMAutocorrelationObject(out_path=out_path)
    autocorr_obj.copy_structure(autocorr_tmp.out_path)
    
    assert set(autocorr_obj.temporal) == set(autocorr_obj.spatial)

    Nfiles = len(files_list)
    for file in files_list:
        # Load autocorrelations of one state
        autocorr_tmp = VMAutocorrelationObject(out_path=file)

        # Compute average
        for key in autocorr_obj.temporal.keys():
            autocorr_obj.temporal[key] += autocorr_tmp.temporal[key] / Nfiles
            autocorr_obj.spatial[key]  += autocorr_tmp.spatial[key]  / Nfiles

    # Get distance and time difference arrays
    autocorr_obj.t_array = autocorr_tmp.t_array
    autocorr_obj.r_array = autocorr_tmp.r_array

    autocorr_obj.save_pickle()


if __name__ == "__main__":
    main()

