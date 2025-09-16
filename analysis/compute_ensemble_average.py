import glob
import pickle
import argparse
import numpy as np
from pathlib import Path

from utils.correlation_object import VMAutocorrelationObject

processed_dir = "data/simulated/processed/"
ensemble_dir  = f"{processed_dir}averages/"
Path(ensemble_dir).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Compute ensemble average of autocorrelation")
    parser.add_argument('dirpath', type=str, help="Path to ensemble directory. Typically 'data/simulated/processed/dir'")
    args = parser.parse_args()

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

