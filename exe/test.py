import os
import sys
from pathlib import Path


arg = sys.argv[1]
parent = Path(arg).parent
print(arg)
print(Path(arg).parent)
print(Path(parent).parent)
print(Path(arg).name)