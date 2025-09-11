import glob
import pickle
import argparse

import numpy as np
from pathlib import Path
from datetime import datetime

from utils.path_handling import decompose_input_path
from utils.correlation_object import VMAutocorrelationObject

ensemble_dir = "data/simulated/obj/averages/"
Path(ensemble_dir).mkdir(parents=True, exist_ok=True)


def compute_average(results):
    """Compute average and standard deviation from the results."""
    # Assuming results are numpy arrays or can be converted to numpy arrays
    stacked_results = np.array(results)
    mean = np.mean(stacked_results, axis=0)
    std_dev = np.std(stacked_results, axis=0)
    return mean, std_dev

def save_ensemble_average(average, std_dev, timestamp):
    """Save the ensemble average and standard deviation to a file."""
    output_file = f"data/obj/ensembles/ensemble_average_{timestamp}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({'mean': average, 'std_dev': std_dev}, f)

def main():
    parser = argparse.ArgumentParser(description="Plot all defined autocorrelations")
    parser.add_argument('filepath', type=str, help="Path to files to plot. Typically 'data/simulated/obj/file'. Filename is on form <'path/to/file'>*.autocorr")
    args = parser.parse_args()

    # List all files
    files_list   = glob.glob(f"{args.filepath}*.autocorr")
    autocorr_tmp = VMAutocorrelationObject(files_list[0])

    # Initialize ensemble correlation object
    autocorr_obj = VMAutocorrelationObject(f"{str(Path(args.filepath).parent)}.autocorr")
    autocorr_obj.copy_structure(autocorr_tmp.path)

    assert set(autocorr_obj.temporal) == set(autocorr_obj.spatial)

    Nfiles = len(files_list)
    for file in files_list:
        # Load autocorrelations of one state
        autocorr_tmp = VMAutocorrelationObject(file)

        # Compute average
        for key in autocorr_obj.temporal.keys():
            print(np.shape(autocorr_obj.temporal[key]), np.shape(autocorr_tmp.temporal[key]))
            autocorr_obj.temporal[key] += autocorr_tmp.temporal[key] / Nfiles

            print(np.shape(autocorr_obj.spatial[key]), np.shape(autocorr_tmp.spatial[key]))
            autocorr_obj.spatial[key]  += autocorr_tmp.spatial[key]  / Nfiles

    # Get distance and time difference arrays
    autocorr_obj.t_array = autocorr_tmp.t_array
    autocorr_obj.r_array = autocorr_tmp.r_array

    autocorr_obj.save_pickle()


if __name__ == "__main__":
    main()

