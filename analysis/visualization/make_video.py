import sys
import shutil
import argparse
import platform
import subprocess
from pathlib import Path

sys.path.append("analysis/utils")
sys.path.append("exe/")

import vm_output_handling as vm_output

from cells.init import movie_sh_fname
from utils.plotting_functions import plot
from utils.exception_handlers import save_snapshot

# to plot on my laptop
import matplotlib
matplotlib.use("Agg")

videos_dir = "data/simulated/videos/"
frames_dir = "data/simulated/frames/"

if platform.node() != 'silja-work':
    print(f"Running simulation from {platform.node()}")
    videos_dir = "../../../../hdd_data/silja/VertexModel_data/simulated/videos/"
    frames_dir = "../../../../hdd_data/silja/VertexModel_data/simulated/frames/"


def main():
    parser = argparse.ArgumentParser(description="Creates video from vm_output")
    parser.add_argument('path',    type=str, help="Defines path to file, typically: data/simulated/raw/dir/file.p.")
    parser.add_argument('--cbar0', type=str, help='How define 0 level of cbar in vm video',    default='average')
    parser.add_argument('--replot', action="store_true")
    # frame_rate
    args = parser.parse_args()

    # set paths
    fname = Path(args.path).parent.stem
    path_to_frames = f"{frames_dir}{fname}"
    path_to_videos = f"{videos_dir}{fname}"

    if args.replot:

        shutil.rmtree(path_to_frames)
        Path(path_to_frames).mkdir(parents=True, exist_ok=True)

        # load vm object
        list_vm, init_vm = vm_output.load(args.path)
        print(list_vm[0].systemSize)

        # outputs
        fig, ax = plot(list_vm[0], fig=None, ax=None, cbar_zero=args.cbar0)

        frame = 1
        for vm in list_vm:
            # plot snapshot
            save_snapshot(vm, fig, ax, path_to_frames, frame, cbar_zero=args.cbar0)
            frame += 1


    # make movie
    subprocess.call([movie_sh_fname,
                    "-d", path_to_frames,
                    "-o", f"{path_to_videos}.mp4",
                    "-p", sys.executable,
                    "-y"])

#     os.system('stty sane')


if __name__ == "__main__":
    main()

