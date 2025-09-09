import json


def load_config(config_path):
    """Load configuration from a JSON file."""

    with open(config_path, 'r') as f:
        return json.load(f)

    
def save_config(config_path, config):
    """Save configuration to a JSON file."""
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)



def update_value(config, key, val):
    """
    Retrieve a value from a nested dictionary given a key.
    
    Parameters:
        config (dict): The nested configuration dictionary.
        key (str): The key to search for.
        
    Returns:
        value: The value associated with the key, or None if the key does not exist.
    """
    for sub_dict in config.values():
        if isinstance(sub_dict, dict) and key in sub_dict:
            sub_dict[key] = val
        
    return None

