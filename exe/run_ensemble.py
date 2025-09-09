import os
import json
import subprocess
import glob
import pickle
from datetime import datetime

# Define paths
config_path = 'data/config/run_configs/'
output_path = 'data/obj/'

def load_configurations(config_path):
    """Load all configuration JSON files from the specified directory."""
    config_files = glob.glob(os.path.join(config_path, '*.json'))
    configs = []
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = json.load(f)
            configs.append((config_file, config))  # Tuple of (filename, config)
    return configs

def run_simulation(config):
    """Run the simulation using the specified configuration."""
    config_file, parameters = config
    # You can pass any command-line arguments needed for your script
    subprocess.run(['python', 'exe/run_simulation.py', config_file], check=True)
    return parameters  # Return parameters to potentially save or log

def save_results(results, timestamp):
    """Save the results (parameters) to a pickle file."""
    output_file = os.path.join(output_path, f'analysis_results_{timestamp}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

def main():
    # Load configurations
    configs = load_configurations(config_path)
    results = []

    # Timestamp for saving results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Run simulations for each configuration
    for config in configs:
        print(f"Running simulation with configuration: {config[0]}")
        parameters = run_simulation(config)
        results.append(parameters)

    # Save the results
    save_results(results, timestamp)
    print(f"All simulations completed. Results saved in: {output_path}")

if __name__ == "__main__":
    main()
