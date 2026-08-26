import os
import json
from datetime import datetime


def load(config_path):
    """Load configuration from a JSON file."""

    with open(config_path, 'r') as f:
        return json.load(f)


    
def save(config_path, config_dict, script_file):
    """Save configuration to a JSON file."""

    config_dict["script"] = script_file
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    os.chmod(config_path, 0o444)



def update_value(config_dict, key, val):
    """
    Retrieve a value from a nested dictionary given a key.
    
    Parameters:
        config (dict): The nested configuration dictionary.
        key (str): The key to search for.
        val: The value associated with the key, or None if the key does not exist.
    """
    val = float(val)

    # Transform to int
    if int(val) == val:
        val = int(val)

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