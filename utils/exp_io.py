import sys
import numpy as np
from pathlib import Path
# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


from ExperimentData import SegmentedCells
from Correlations import ExperimentalAutocorrelations
from paths_config import EXP_RAW_DIR, EXP_PROC_DIR, EXP_DATASETS_IDS



def load_full_experiments(DATASET_IDS=EXP_DATASETS_IDS):
    """ Load full dataset from all experiments in EXP_DATASETS_IDS """

    experiments = []
    for ID in DATASET_IDS:
        cellprop = SegmentedCells(f"{EXP_RAW_DIR}/{ID}_cells.p")
        experiments.append(cellprop)

    return experiments



def load_all_correlations(DATASET_IDS=EXP_DATASETS_IDS):
    """ Load all correlations of all experiments """

    experiments = []
    for ID in DATASET_IDS:
        correlation = ExperimentalAutocorrelations(f"{EXP_PROC_DIR}/{ID}_cells.autocorr")
        experiments.append(correlation)

    return experiments
