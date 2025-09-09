import os
import pickle
import numpy as np
from datetime import datetime

def load_results(results_path):
    """Load pickle results from the specified directory."""
    results = []
    for filename in os.listdir(results_path):
        if filename.endswith('.pkl'):
            with open(os.path.join(results_path, filename), 'rb') as f:
                results.append(pickle.load(f))
    return results

def compute_average(results):
    """Compute average and standard deviation from the results."""
    # Assuming results are numpy arrays or can be converted to numpy arrays
    stacked_results = np.array(results)
    mean = np.mean(stacked_results, axis=0)
    std_dev = np.std(stacked_results, axis=0)
    return mean, std_dev

def save_ensemble_average(average, std_dev, timestamp):
    """Save the ensemble average and standard deviation to a file."""
    output_file = f"data/obj/ensembles/ensemble_average_{timestamp}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({'mean': average, 'std_dev': std_dev}, f)

def main():
    results_path = 'data/obj/ensembles/ensemble_YYYYMMDD_HHMM/run_specific_results/'  # Adjust accordingly
    results = load_results(results_path)
    
    average, std_dev = compute_average(results)
    
    # Get current timestamp for naming the output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_ensemble_average(average, std_dev, timestamp)

if __name__ == "__main__":
    main()
