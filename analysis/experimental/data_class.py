import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append("code/preprocessing/utils")
sys.path.append("code/analysis/")
import masked_correlation_functions as compute


class SegmentationData:
    def __init__(self):
        self.date = datetime.today().strftime('%Y/%m/%d_%H:%M')


    def load(self, path):
        """
        Loads the state from a pickle file.

        Parameters:
        - path: path to pickle to load.
        """
        
        # Load pickle
        with open(f"{path}", 'rb') as f:
            state = pickle.load(f)
        
        # Update object
        self.x = state.get('x', {})
        self.y = state.get('y', {})
        self.h = state.get('h', {})
        self.A = state.get('A', {})

        self.dx = state.get('dx', {})
        self.dy = state.get('dy', {})

        self.label  = state.get('label', {})
        self.aminor = state.get('aminor', {})
        self.amajor = state.get('amajor', {})

        self.density = state.get('density', {})

        try:
            self.n = state.get('n', {})
        except:
            pass

        print(f"State loaded from {path}.")


    def save(self, path):
        """ Saves object as pickle"""

        # Prepare state dictionary to save
        state = {
            'x': self.x,
            'y': self.y,
            'h': self.h,
            'A': self.A,
            'dx': self.dx,
            'dy': self.dy,
            'label': self.x,
            'aminor': self.aminor,
            'amajor': self.amajor, 
            'density': self.density
        }

        try:
            state['n'] = self.n
        except:
            pass
        
        # Save
        with open(f"{path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {path}")


    def add(self, param, value):
        if param == "density":
            self.density = value

    def add_density(self):
        Ncells = np.ma.sum(~self.A.mask, axis=1)
        Acells = np.ma.sum(self.A, axis=1)
        self.density = 10**6 * Ncells / Acells


class VariationData:
    def __init__(self, path):

        self.datatype = "empty"
        self.path = path

        if os.path.isfile(self.path):
            self.load()



    def add_unbinned_data(self, density, height, variation, measure):
        self.datatype = "unbinned"
        self.density  = density

        if measure == "pixel":
            self.var_pixel = variation
            self.h_pixel   = height

        elif measure == "disk":
            self.var_disk = variation
            self.h_disk   = height
        
        elif measure == "cell":
            self.var_cell = variation
            self.h_cell   = height



    def add_binned_data(self, density, variation, svariation, measure):
        self.datatype = "binned"
        self.density  = density

        if measure == "pixel":
            self.var_pixel  = variation
            self.svar_pixel = svariation
            # self.h_pixel  = height
            # self.sh_pixel = sheight

        elif measure == "disk":
            self.var_disk  = variation
            self.svar_disk = svariation
            # self.h_disk  = height
            # self.sh_disk = sheight
        
        elif measure == "cell":
            self.var_cell  = variation
            self.svar_cell = svariation
            # self.h_cell  = height
            # self.sh_cell = sheight



    def load(self):
        # Load pickle
        with open(f"{self.path}", 'rb') as f:
            state = pickle.load(f)
        
        # Update object
        self.density  = state.get('density', {})
        self.datatype = state.get('datatype', {})

        self.var_pixel = state.get('var_pixel', {})
        self.var_disk  = state.get('var_disk', {})
        self.var_cell  = state.get('var_cell', {})

        if self.datatype == "binned":
            self.svar_pixel = state.get('svar_pixel', {})
            self.svar_disk  = state.get('svar_disk', {})
            self.svar_cell  = state.get('svar_cell', {})
            self.sh_pixel = state.get('sh_pixel', {})
            self.sh_disk  = state.get('sh_disk', {})
            self.sh_cell  = state.get('sh_cell', {})

        if self.datatype == "unbinned":
            self.h_pixel = state.get('h_pixel', {})
            self.h_disk  = state.get('h_disk', {})
            self.h_cell  = state.get('h_cell', {})

        print(f"State loaded from {self.path}.")



    def save(self):
        """ Saves object as pickle"""

        # Prepare state dictionary to save
        state = {
            'density':   self.density,
            'datatype':  self.datatype,
            'var_pixel': self.var_pixel,
            'var_disk':  self.var_disk,
            'var_cell':  self.var_cell,
            }
        
        if self.datatype == "binned":
            state['svar_pixel'] = self.svar_pixel
            state['svar_disk']  = self.svar_disk
            state['svar_cell']  = self.svar_cell
            # state['sh_pixel'] = self.sh_pixel
            # state['sh_disk']  = self.sh_disk
            # state['sh_cell']  = self.sh_cell
        
        if self.datatype == "unbinned":
            state['h_pixel'] = self.h_pixel
            state['h_disk']  = self.h_disk
            state['h_cell']  = self.h_cell

        # Save
        with open(f"{self.path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {self.path}")



    def bin_data(self, variable, bin_size=100):

        min_bin = int(np.min(self.density) / bin_size) * bin_size
        max_bin = int(np.max(self.density) / bin_size) * bin_size
        bins = np.arange(min_bin, max_bin + bin_size, bin_size)

        mean_variable = np.zeros(len(bins) - 1)
        std_variable  = np.zeros(len(bins) - 1)
        counts        = np.zeros(len(bins) - 1)

        density_idx = np.digitize(self.density, bins)

        for i in range(1, len(bins)):
            idx_in_bin = np.where(density_idx == i)[0]
            counts[i-1] = len(idx_in_bin)

            if counts[i-1] == 0:
                #mean_variable[i-1] = np.nan
                #std_variable[i-1]  = np.nan
                continue

            mean_variable[i-1] = np.mean(variable[idx_in_bin])
            if counts[i-1] > 1:
                std_variable[i-1]  = np.std(variable[idx_in_bin], ddof=1)



        output = {
            "density_bins": bins,
            "mean": mean_variable,
            "std": std_variable
            #"N_in_bin": counts
        }

        return output
            


class AutocorrelationData:
    def __init__(self, path):

        self.path = path

        if os.path.isfile(self.path):
            self.load()

        else:
            self.temporal = {}
            self.spatial  = {}
            self.t_array  = {}
            self.r_array  = {}
            self.log = {'t': {},
                        'r': {}}



    def load(self):
        """
        Loads the state from a pickle file.

        Parameters:
        - path: path to pickle to load.
        """
        
        # Load pickle
        with open(f"{self.path}", 'rb') as f:
            state = pickle.load(f)
        
        # Update object
        self.temporal = state.get('temporal', {})
        self.spatial  = state.get('spatial', {})
        self.t_array  = state.get('t_array', {})
        self.r_array  = state.get('r_array', {})
        self.log      = state.get('log', {})

        print(f"State loaded from {self.path}.")


    
    # def copy_structure(self, struct_path, path_addition=''):
    #     """
    #     Copies structure of object at struct_path
    #     """

    #     # Load pickle
    #     with open(f"{path_addition}{struct_path}", 'rb') as f:
    #         structure = pickle.load(f)

    #     self.temporal = {key: np.ma.zeros_like(value) for key, value in structure.pop('temporal').items()}
    #     self.spatial  = {key: np.ma.zeros_like(value) for key, value in structure.pop('spatial').items()}
    #     self.t_array  = {key: np.ma.zeros_like(value) for key, value in structure.pop('t_array').items()}
    #     self.r_array  = {key: np.ma.zeros_like(value) for key, value in structure.pop('r_array').items()}

    #     print(f"Copied structure of {struct_path}")



    def save(self):
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
        with open(f"{self.path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {self.path}")



    def compute_spatial(self, positions, variable, variable_name, dr, r_max, t_avrg=False, overwrite=False):
        """ Computes spatial autocorrelation """

        # Check if correlation exists
        if not overwrite:
            if variable_name in self.spatial.keys():
                print(f"Spatial autocorrelation of {variable_name} already exists.")
                return

        # Compute autocorrelation
        fmax = variable.shape[-2]

        Cr = compute.general_spatial_correlation(positions[0][:fmax], positions[1][:fmax], variable,
                                                 dr=dr, r_max=r_max, t_avrg=t_avrg)

        # Update object
        self.spatial[variable_name]  = Cr['C_norm']#.compressed()
        self.r_array[variable_name]  = Cr['r_bin_centers']#.compressed()
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



    
    def bin_data(self, bin_size=200):

        min_bin = int(np.min(self.density) / bin_size) * bin_size
        max_bin = int(np.max(self.density) / bin_size) * bin_size
        bins = np.arange(min_bin, max_bin + bin_size, bin_size)

        mean_variable = np.zeros(len(bins) - 1)
        std_variable  = np.zeros(len(bins) - 1)
        counts        = np.zeros(len(bins) - 1)

        density_idx = np.digitize(self.density, bins)

        for i in range(1, len(bins)):
            idx_in_bin = np.where(density_idx == i)[0]
            counts[i-1] = len(idx_in_bin)

            if counts[i-1] == 0:
                #mean_variable[i-1] = np.nan
                #std_variable[i-1]  = np.nan
                continue

            mean_variable[i-1] = np.mean(variable[idx_in_bin])
            if counts[i-1] > 1:
                std_variable[i-1]  = np.std(variable[idx_in_bin], ddof=1)



        output = {
            "density_bins": bins,
            "mean": mean_variable,
            "std": std_variable
            #"N_in_bin": counts
        }

        return output