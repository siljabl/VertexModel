import os
import sys
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from cells.bind import getPercentageKeptNeighbours

sys.path.append("analysis/utils")
import vm_output_handling as vm_output

parser = argparse.ArgumentParser(description="Run several runs")
parser.add_argument('dirs',   nargs='*',  help="directories")
args = parser.parse_args()


# directories = os.listdir(args.dirname)

# # dir_path = "data/simulated/raw/nodivision_20250919_N30_L64_Lambda100_v0100_taup400/*"

bond_breaking = np.zeros([len(args.dirs),900])

d = 0
for dir in tqdm(args.dirs):
    for path in glob(dir+"/*"):

        list_vm, init_vm = vm_output.load(path)
        print(len(list_vm))

        for i in range(len(list_vm)):
            neighbour_dict = getPercentageKeptNeighbours(list_vm[0], list_vm[i])
            bond_breaking[d,i] += np.mean(list(neighbour_dict.values()))
    
    d += 1



fig = plt.figure()
plt.plot(bond_breaking[0])
plt.savefig("Bond_breaking_correlation.png")