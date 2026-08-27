import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import utils.vm_output_handling as vm_output

from paths_config             import SIM_RAW_DIR, SIM_FRAMES_DIR
from cells.init               import movie_sh_fname
from utils.vm_plotting        import plot_frame
from utils.exception_handlers import save_frame

VIDEOS_DIR = Path("movies")
data_dir   = "../../../../hdd_data/silja/VertexModel/sim/"


def main():
    parser = argparse.ArgumentParser(description="Creates video from vm_output")
    parser.add_argument('config_path',       type=str,                        help="Defines path to file, typically: configs/constant_volume/parameters/N36_seed0.json")
    parser.add_argument('--cbar0',           type=str,   default='absolute',  help='How define 0 level of cbar in vm video')
    parser.add_argument('--fps',             type=int,   default="10",        help="Frame rate (frames per second) for the output video")
    parser.add_argument('--hmax',            type=float, default=14)
    parser.add_argument('-o', '--overwrite', action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Infer dirname and runname from config path:
    # configs/<mode>/<dirname>/<runname>.json
    mode    = config_path.parent.parent.name
    dirname = config_path.parent.name
    runname = config_path.stem

    # Paths for data, frames, and video
    data_path    = Path(SIM_RAW_DIR)    / mode / dirname / f"{runname}.p"
    frames_dir   = Path(SIM_FRAMES_DIR) / mode / dirname / runname
    videos_dir   = Path(VIDEOS_DIR)     / mode / dirname
    videos_dir.mkdir(parents=True, exist_ok=True)


    if args.overwrite:
        # Clear and recreate frames directory
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load vm objects (list of frames) — adjust init_time if needed
        list_vm, init_vm = vm_output.load(data_path, init_time=100)
        print(f"Loaded {len(list_vm)} frames from {data_path}")

        # Initialise plot with first frame
        fig, ax = plot_frame(list_vm[0], fig=None, ax=None, cbar_zero=args.cbar0, hmax=args.hmax)

        frame = 0
        for vm in tqdm(list_vm, desc="Saving frames"):
            # plot snapshot
            save_frame(vm, fig, ax, frames_dir, frame, cbar_zero=args.cbar0, hmax=args.hmax)
            frame += 1

    # Make movie from FRAMES_DIR
    subprocess.call([
        movie_sh_fname,
        "-d", str(frames_dir),
        "-o", str(videos_dir / runname),
        "-p", sys.executable,
        "-y",
        "-r", str(args.fps),
    ])

if __name__ == "__main__":
    main()

