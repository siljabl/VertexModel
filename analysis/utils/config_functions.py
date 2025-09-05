import json

def load(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_value(config, key):
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
            return sub_dict[key]
        
    return None  # Return None if key is not found