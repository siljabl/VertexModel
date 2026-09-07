import os
import json


def load(config_path):
    """Load configuration from a JSON file."""

    with open(config_path, 'r') as f:
        return json.load(f)


    
def save(config_path, config_dict, script_file):
    """Save configuration to a JSON file."""

    config_dict["script"] = script_file
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    # Make config read-only (owner/group/world) after saving
    os.chmod(config_path, 0o444)



def update_value(config_dict, key, val):
    """
    Retrieve a value from a nested dictionary given a key.
    
    Parameters:
        config (dict): The nested configuration dictionary.
        key (str): The key to search for.
        val: The value associated with the key, or None if the key does not exist.
    """

    # Cast to float
    val = float(val)

    # Cast to int
    if int(val) == val:
        val = int(val)

    # Update all sub-dicts that contain this key
    for sub_dict in config_dict.values():
        if isinstance(sub_dict, dict) and key in sub_dict:
            sub_dict[key] = val
        
    return None



def get_value(config_dict, key):
    """
    Retrieve a value from a nested dictionary given a key.
    
    Parameters:
        config (dict): The nested configuration dictionary.
        key (str): The key to search for.
        
    Returns:
        value: The value associated with the key, or None if the key does not exist.
    """
    for sub_dict in config_dict.values():
        if isinstance(sub_dict, dict) and key in sub_dict:
            return sub_dict[key]
        
    return None  # Return None if key is not found



def create_run_fname(config):
    """
    Create filepath for simulation output
    """

    N     = config['simulation']['Nvertices']
    seed  = config['simulation']['seed']

    gamma = int(config['physics']['gamma'])
    v0    = int(config['physics']['v0'])
    taup  = int(config['physics']['taup'] * 100)
    eta   = int(config['physics']['eta'] * 100)

    dirname = f"gamma{gamma}_v0{v0}_taup{taup}_eta{eta}"
    fname   = f"N{N}_seed{seed}"

    return dirname, fname