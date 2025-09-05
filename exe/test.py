import os
import sys
from pathlib import Path

data_dir   = "data/simulated/"

arg = sys.argv[1]
print(arg.split(data_dir)[-1])