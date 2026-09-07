import os
import sys
import json
import copy
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.append("utils/")

from paths_config import EXP_RAW_DIR, EXP_PROC_DIR, SIM_RAW_DIR, FRAME_TO_H, EXP_DATASETS_IDS

import cells.bind
import ExperimentData
import vm_io as vm_io
import vm_observables as get_vm
import exp_observables as get_exp

from ExperimentData import SegmentedCells
from matrix_preprocessing import cell_variations
from Correlations import ExperimentalAutocorrelations, SimulationAutocorrelations, compute_cell_correlation



def is_simulation_path(path):
    """
    Heuristics:
    - If there is a JSON config file with the same stem → simulation.
    - If the path is under SIM_RAW_DIR → simulation.
    Otherwise, treat as experimental.
    """

    # Same-dir config: seed_00123.p + seed_00123.json
    cfg_candidate = path.with_suffix(".json")
    if cfg_candidate.is_file():
        return True

    # Check known root directories
    sim_root  = Path(SIM_RAW_DIR).resolve()
    exp_raw   = Path(EXP_RAW_DIR).resolve()
    exp_proc  = Path(EXP_PROC_DIR).resolve()
    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path

    if sim_root in resolved.parents:
        return True
    if exp_raw in resolved.parents or exp_proc in resolved.parents:
        return False

    # Fallback: treat as experimental if nothing matches
    return False



def prepare_parameter_map(data, args):

    is_vm   = isinstance(data, list) and isinstance(data[0], cells.bind.VertexModel)
    is_exp  = isinstance(data, ExperimentData.SegmentedCells)
    assert is_vm or is_exp, "data must be list of VertexModel or SegmentedCells"

    # Define mean variable axis
    if args.mean_var == 'r':
        mean_axis = 1
    elif args.mean_var == 'cell': 
        mean_axis = 0
    else:
        raise ValueError("mean_var must be 'r' or 'cell'")

    if is_vm:
        positions = get_vm.cell_positions(data)
        parameter_map = {
            'hh': (positions,        cell_variations(get_vm.cell_heights(data), mean_axis)),
            'AA': (positions,        cell_variations(get_vm.cell_areas(data),  mean_axis)),
            'VV': (positions,        cell_variations(get_vm.cell_volumes(data), mean_axis)),
            'vv': (positions[:, :-1], cell_variations(get_vm.cell_displacement_vectors(data), mean_axis)),
        }
    else:
        positions = np.ma.array([data.x, data.y])
        parameter_map = {
            'hh': (positions,        cell_variations(get_exp.get_attribute(data, 'h', args.dt), mean_axis)),
            'AA': (positions,        cell_variations(get_exp.get_attribute(data, 'A', args.dt), mean_axis)),
            'VV': (positions,        cell_variations(get_exp.get_attribute(data, 'V', args.dt), mean_axis)),
            'vv': (positions[:, :-1], cell_variations(get_exp.get_attribute(data, 'v_vec', args.dt), mean_axis)),
        }

    return parameter_map



def process_one_path(path, base_args):
    """
    Worker function: load data for a single path, build parameter_map,
    compute correlations, and save.
    """
    # Make a per-path copy of args so we can set dt without touching others
    args = copy.copy(base_args)

    # Mapping between experimental ids and dt
    ID_TO_F2H = dict(zip(EXP_DATASETS_IDS, FRAME_TO_H))

    if is_simulation_path(path):
        # ---- Simulation branch ----
        mode   = path.parent.parent.name  # e.g. 'mode'
        params = path.parent.name         # e.g. 'params'
        seed   = Path(str(path.stem) + ".p")

        # Directory with simulation .p files
        sim_path = Path(SIM_RAW_DIR) / mode / params / seed

        data, _      = vm_io.load_simulation(sim_path)
        autocorr_obj = SimulationAutocorrelations(in_path=str(sim_path))

        # dt from simulation
        args.dt = data[1].time - data[0].time

    else:
        # ---- Experimental branch ----
        dataset_id   = path.stem.replace("_cells", "")
        data         = SegmentedCells(f"{EXP_RAW_DIR}/{dataset_id}_cells.p")
        autocorr_obj = ExperimentalAutocorrelations(f"{EXP_PROC_DIR}/{dataset_id}_cells.autocorr")

        # dt from mapping
        try:
            args.dt = ID_TO_F2H[dataset_id]
        except:
            pass

    # Create parameter map
    parameter_map = prepare_parameter_map(data, args)

    # Compute correlations
    compute_cell_correlation(autocorr_obj, parameter_map, args)



def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("path",          type=str,   help="Path to dataset, as data/experimental/processed/dataset")
    parser.add_argument('-p', '--param', type=str,   help="Parameter to plot correlation of (varvar)",                          default="all")
    parser.add_argument('-v', '--var',   type=str,   help="Correlation variable (t or r)",                                      default="all")
    parser.add_argument('-o','--overwrite',          help="Overwrite previous computations",                                    action='store_true')
    parser.add_argument('--dr',          type=float, help="Spatial step size (float)",                                          default=40)
    parser.add_argument('--dt',          type=float, help="Spatial step size (float)",                                          default=1 / 12)
    parser.add_argument('--rfrac',       type=float, help="Max distance to compute correlation for (float)",                    default=0.7)
    parser.add_argument('--tfrac',       type=float, help="Fraction of total duration to compute correlation for (float)",      default=0.9)
    parser.add_argument('--mean_var',    type=str,   help="Variable to take mean over in <x - <x>_var> (r or cell). Default: r",  default='r')
    parser.add_argument('--t_avrg',                  help="Take average over all starting times (if steady state)", action="store_true")
    args = parser.parse_args()

    # Path decomposition
    input_path = Path(args.path)

    if input_path.is_dir():
        print("input_path is dir")
        run_paths = sorted(input_path.glob("*.json"))
        if len(run_paths) == 0:
            run_paths = sorted(input_path.glob("*.p"))
        print(run_paths)
    else:
        # Single file
        if not input_path.is_file():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        run_paths = [input_path]

    # Decide serial vs parallel based on number of paths
    if len(run_paths) == 1:
        # Just one file: run serially
        process_one_path(run_paths[0], args)
    else:
        # Multiple files: parallelise
        Npool = min(len(run_paths), os.cpu_count() or 1)
        with Pool(processes=Npool) as pool:
            pool.starmap(
                process_one_path,
                [(path, args) for path in run_paths]
            )



if __name__ == "__main__":
    main()
