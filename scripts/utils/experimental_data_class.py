import os
import pickle
import numpy as np
from datetime import datetime

# sys.path.append("code/preprocessing/utils")
# sys.path.append("code/scripts/")


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




class AutocorrelationData:
    def __init__(self, path):

        self.path = path

        if os.path.isfile(self.path):
            self.load()

        else:
            self.temporal_cell = {}
            self.temporal = {}
            self.spatial  = {}
            self.t_array  = {}
            self.r_array  = {}
            self.density  = {}
            self.log = {'t_cell': {},
                        't': {},
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
        self.temporal_cell = state.get('temporal_cell', {})
        self.temporal = state.get('temporal', {})
        self.spatial  = state.get('spatial', {})
        self.t_array  = state.get('t_array', {})
        self.r_array  = state.get('r_array', {})
        self.density  = state.get('density', {})
        self.log      = state.get('log', {})

        print(f"State loaded from {self.path}.")



    def save(self):
        """ Saves object as pickle"""

        # Prepare state dictionary to save
        state = {
            'temporal_cell': self.temporal_cell,
            'temporal': self.temporal,
            'spatial':  self.spatial,
            't_array':  self.t_array,
            'r_array':  self.r_array,
            'density':  self.density,
            'log':      self.log
        }
        
        # Save
        with open(f"{self.path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {self.path}")


