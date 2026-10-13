import os
import json


def load(config_path):
    """Load configuration from a JSON file."""

    with open(config_path, 'r') as f:
        return json.load(f)


    
def save(config_path, config_dict, mode):
    """Save configuration to a JSON file."""

    config_dict["mode"] = mode
    
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



def create_run_fname(config, mode):
    """
    Create filepath for simulation output
    """

    rho   = config["cells"]["target_density"]
    N     = config['simulation']['n_vertices']
    seed  = config['simulation']['seed']

    if mode == "constant_volume":
        fname = f"N{N}_seed{seed}"

    else:
        fname = f"rho{rho}_seed{seed}"

    if mode in ("constant_volume", "emergent_volume"):
        gamma = int(config['forces']['gamma'])
        v0    = int(config['forces']['v0'])
        taup  = int(config['forces']['taup'] * 100)
        eta   = int(config['forces']['eta'] * 100)

        dirname = f"gamma{gamma}_v0{v0}_taup{taup}_eta{eta}"

    
    elif mode in ("death_and_division", "growing_volume"):
        gamma = int(config['forces']['gamma'])
        v0    = int(config['forces']['v0'])
        g     = int(config['growth']['g'])
        kdiv  = int(config['growth']['k_div'] * 1e6)
        Vdiv  = int(config['growth']['V_div'])


        dirname = f"g{g}_kdiv{kdiv}_Vdiv{Vdiv}"


    return dirname, fname