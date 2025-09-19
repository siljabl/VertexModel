import os
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import utils.correlation_computations as compute

data_dir = "data/simulated/raw/"
obj_dir  = "data/simulated/processed/"

if platform.node() != 'silja-work':
    data_dir = "../../../../hdd_data/silja/VertexModel_data/simulated/raw/"
    obj_dir  = "../../../../hdd_data/silja/VertexModel_data/simulated/processed/"

                    

class VMAutocorrelationObject:
    def __init__(self, in_path=None, out_path=None, path_addition=''):
        """
        Initializes the autocorrelation object and checks if pickle exists.

        Parameters:
        - filname: name of simulation data (and pickle)
        - path_addition: path that redirects to the project folder. Mainly for running in notebooks
        """
        assert in_path != None or out_path != None, 'Must provide either in_path or out_path'

        if in_path != None:
            filename = f"{Path(in_path.split(data_dir)[-1]).with_suffix('')}"
        elif out_path != None:
            filename = f"{Path(out_path.split(obj_dir)[-1]).with_suffix('')}"
        
        self.in_path  = f"{data_dir}{filename}.p"
        self.out_path = f"{obj_dir}{filename}.autocorr"

        self.temporal = {}
        self.spatial  = {}
        self.t_array  = {}
        self.r_array  = {}
        self.log = {'t': {},
                    'r': {}}

        # Check if the state file exists and load it if it does
        if os.path.exists(f"{path_addition}{self.out_path}"):
            self.load_state(path_addition=path_addition)
        else:
            print(f"No saved state file found at {path_addition}{self.out_path}. Starting fresh with provided data.")



    def load_state(self, path_addition=''):
        """
        Loads the state from a pickle file.

        Parameters:
        - path: path to pickle to load.
        """
        
        # Load pickle
        with open(f"{path_addition}{self.out_path}", 'rb') as f:
            state = pickle.load(f)
        
        # Update object
        self.temporal = state.get('temporal', {})
        self.spatial  = state.get('spatial', {})
        self.t_array  = state.get('t_array', {})
        self.r_array  = state.get('r_array', {})
        self.log      = state.get('log', {})

        print(f"State loaded from {path_addition}{self.out_path}.")


    
    def copy_structure(self, struct_path, path_addition=''):
        """
        Copies structure of object at struct_path
        """

        # Load pickle
        with open(f"{path_addition}{struct_path}", 'rb') as f:
            structure = pickle.load(f)

        self.temporal = {key: np.ma.zeros_like(value) for key, value in structure.pop('temporal').items()}
        self.spatial  = {key: np.ma.zeros_like(value) for key, value in structure.pop('spatial').items()}
        self.t_array  = {key: np.ma.zeros_like(value) for key, value in structure.pop('t_array').items()}
        self.r_array  = {key: np.ma.zeros_like(value) for key, value in structure.pop('r_array').items()}

        print(f"Copied structure of {struct_path}")



    def save_pickle(self, path_addition=''):
        """ Saves object as pickle"""

        # Prepare state dictionary to save
        state = {
            'temporal': self.temporal,
            'spatial':  self.spatial,
            't_array':  self.t_array,
            'r_array':  self.r_array,
            'log':      self.log
        }
        
        # Save
        with open(f"{path_addition}{self.out_path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {path_addition}{self.out_path}")



    def compute_spatial(self, positions, variable, variable_name, dr, r_max, t_avrg=False, overwrite=False):
        """ Computes spatial autocorrelation """

        # Check if correlation exists
        if not overwrite:
            if variable_name in self.spatial.keys():
                print(f"Spatial autocorrelation of {variable_name} already exists.")
                return

        # Compute autocorrelation
        Cr = compute.general_spatial_correlation(positions[:,:,0], positions[:,:,1], variable,
                                                 dr=dr, r_max=r_max, t_avrg=t_avrg)

        # Update object
        self.spatial[variable_name]  = Cr['C_norm'].compressed()
        self.r_array[variable_name]  = Cr['r_bin_centers'].compressed()
        self.log['r'][variable_name] = datetime.today().strftime('%Y/%m/%d_%H:%M')



    def compute_temporal(self, variable, variable_name, t_max, t_avrg=False, overwrite=False):
        """ Computes temporal autocorrelation """

        # Check if correlation exists
        if not overwrite:
            if variable_name in self.temporal.keys():
                print(f"Temporal autocorrelation of {variable_name} already exists.")
                return

        # Compute autocorrelation    
        Ct = compute.general_temporal_correlation(variable, t_max=t_max, t_avrg=t_avrg)

        # Update object
        self.temporal[variable_name] = Ct['C_norm']
        self.t_array[variable_name]  = np.arange(t_max)
        self.log['t'][variable_name] = datetime.today().strftime('%Y/%m/%d_%H:%M')
