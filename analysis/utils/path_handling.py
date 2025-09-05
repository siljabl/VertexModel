from pathlib import Path

"""
This file should probably be replaced by the use of Path
"""


def decompose_input_path(filepath, dirpath):
    """ 
    Takes input path and returns filename and relative path from dir to file 

    Parameters:
    - filepath: path to file, as  path/to/dir/relative/path/to/file.ext
    - dirpath:  path to directory, as path/to/dir/

    Returns:
    - file: filename without extension (if any)
    - relative_parent: path between dir and file
    """

    # Decompose input path
    relative_path = filepath.split(dirpath)[-1]
    file = Path(relative_path).stem
    relative_parent = f"{Path(relative_path).parent}/"

    return relative_parent, file
