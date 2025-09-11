import glob
import pickle
import argparse

import numpy as np
from pathlib import Path
from datetime import datetime

from utils.path_handling import decompose_input_path
from utils.correlation_object import VMAutocorrelationObject

obj_dir = "data/simulated/obj/"
ensemble_dir = "data/simulated/obj/averages/"
Path(ensemble_dir).mkdir(parents=True, exist_ok=True)


def compute_average(results):
    """Compute average and standard deviation from the results."""
    # Assuming results are numpy arrays or can be converted to numpy arrays
    stacked_results = np.array(results)
    mean = np.mean(stacked_results, axis=0)
    std_dev = np.std(stacked_results, axis=0)
    return mean, std_dev



def main():
    parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
    parser.add_argument('filepath', type=str, help="Path to files to plot. Typically 'data/simulated/obj/file'. Filename is on form <'path/to/file'>*.autocorr")
    args = parser.parse_args()

    # List all files
    files_list   = glob.glob(f"{args.filepath}*.autocorr")
    autocorr_tmp = VMAutocorrelationObject(out_path=files_list[0])

    # Initialize ensemble correlation object
    out_path = f"{ensemble_dir}{Path(args.filepath).stem}"
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

