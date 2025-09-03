import os
import pickle
import numpy as np
from datetime import datetime

import utils.correlation_computations as compute

class VMAutocorrelationObject:
    def __init__(self, filename, path_addition=''):
        """
        Initializes the autocorrelation object and checks if pickle exists.

        Parameters:
        - filname: name of simulation data (and pickle)
        - path_addition: path that redirects to the project folder. Mainly for running in notebooks
        """
        assert filename is not None
        
        root, _ = os.path.splitext(filename)
        
        self.fname = filename
        self.path  = f"data/simulated/obj/autocorrelation_{root}.obj"

        self.temporal = {}
        self.spatial  = {}
        self.t_array  = {}
        self.r_array  = {}
        self.log = {'t': {},
                    'r': {}}

        # Check if the state file exists and load it if it does
        if os.path.exists(f"{path_addition}{self.path}"):
            self.load_state(path_addition=path_addition)
        else:
            print(f"No saved state file found at {path_addition}{self.path}. Starting fresh with provided data.")



    def load_state(self, path_addition=''):
        """
        Loads the state from a pickle file.

        Parameters:
        - path: path to pickle to load.
        """
        
        with open(f"{path_addition}{self.path}", 'rb') as f:
            state = pickle.load(f)
        
        # verify that loading correct file
        assert self.fname == state.get('fname', '')

        self.temporal = state.get('temporal', {})
        self.spatial  = state.get('spatial', {})
        self.t_array  = state.get('t_array', {})
        self.r_array  = state.get('r_array', {})
        self.log      = state.get('log', {})

        print(f"State loaded from {path_addition}{self.path}.")



    def save_pickle(self, path_addition=''):
        """ Saves object as pickle"""
         # Prepare state dictionary to save
        state = {
            'fname':    self.fname,
            'temporal': self.temporal,
            'spatial':  self.spatial,
            't_array':  self.t_array,
            'r_array':  self.r_array,
            'log':      self.log
        }
        
        with open(f"{path_addition}{self.path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {path_addition}{self.path}")



    def compute_spatial(self, positions, variable, dr, r_max, variable_name, t_avrg=False):
        """ Computes spatial autocorrelation """

        Cr = compute.general_spatial_correlation(positions[:,:,0], positions[:,:,1], variable,
                                                 dr=dr, r_max=r_max, t_avrg=t_avrg)

        self.spatial[variable_name]  = Cr['C_norm'].compressed()
        self.r_array[variable_name]  = Cr['r_bin_centers'].compressed()
        self.log['r'][variable_name] = datetime.today().strftime('%Y/%m/%d_%H:%M')

