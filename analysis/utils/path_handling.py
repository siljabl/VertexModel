from pathlib import Path


# data_dir   = "data/simulated/raw/"
# config_dir = "data/simulated/configs/"

def decompose_input_path(filepath, data_dir):

    # Decompose input path
    path_tail = filepath.split(data_dir)[-1]
    filename  = Path(path_tail).name
    parent    = f"{Path(path_tail).parent}/"

    return parent, filename