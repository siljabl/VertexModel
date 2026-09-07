import os
import pickle
import numpy as np
from datetime import datetime



class SegmentedCells:
    """ Container for segmented cell observables from experimental data. """



    def __init__(self, path):
        # Timestamp for this object
        self.date = datetime.today().strftime('%Y/%m/%d_%H:%M')
        self.path = path

        # If a pickle exists at path, immediately load it
        if os.path.isfile(self.path):
            self.load(path)



    def load(self, path):
        """
        Loads the state from a pickle file.

        Parameters:
        - path: path to pickle to load.
        """
        
        # Load pickle
        with open(f"{path}", 'rb') as f:
            state = pickle.load(f)
        
        # Restore core observables; use {} as default if missing
        self.x = state.get('x', {})
        self.y = state.get('y', {})
        self.h = state.get('h', {})
        self.A = state.get('A', {})

        self.dx = state.get('dx', {})
        self.dy = state.get('dy', {})

        self.label  = state.get('label', {})
        self.aminor = state.get('aminor', {})
        self.amajor = state.get('amajor', {})
        self.theta  = state.get('theta', {})
        self.ecc    = state.get('ecc', {})

        self.density = state.get('density', {})

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
            'label': self.label,
            'aminor': self.aminor,
            'amajor': self.amajor, 
            'theta':  self.theta, 
            'ecc': self.ecc, 
            'density': self.density
        }

        
        # Save
        with open(f"{path}", 'wb') as f:
            pickle.dump(state, f)

        print(f"State saved to {path}")



    def filter_by_density(self, density_mask):
        """
        Keep only frames (timepoints) where density_mask is True.

        Parameters
        ----------
        density_mask : array-like of bool, shape (T,)
            Boolean mask over the time axis (frames). True = keep, False = drop.
        """
        density_mask = np.asarray(density_mask, dtype=bool)

        # Basic sanity check against current number of frames
        T = self.x.shape[0]
        if density_mask.shape[0] != T:
            raise ValueError(f"density_mask length {density_mask.shape[0]} != number of frames {T}")

        # Filter all per-frame arrays along axis 0
        self.x      = self.x[density_mask]
        self.y      = self.y[density_mask]
        self.h      = self.h[density_mask]
        self.A      = self.A[density_mask]
        # dx/dy are one frame shorter than x/y
        self.dx     = self.dx[density_mask[:-1]]
        self.dy     = self.dy[density_mask[:-1]]
        self.label  = self.label[density_mask]
        self.aminor = self.aminor[density_mask]
        self.amajor = self.amajor[density_mask]
        self.theta  = self.theta[density_mask]
        self.ecc    = self.ecc[density_mask]


        # Recompute density from area
        self.density = 10 ** 6 / np.ma.mean(self.A, axis=1)



    def drop_empty_cells(self):
        """
        Remove cells (columns) that are fully masked in A.
        Updates all per-cell observables and dx/dy.
        """
        # Use area as reference for "empty" cells
        fully_masked_cells = np.all(self.A.mask, axis=0)  # shape (Ncell,)
        cell_mask = ~fully_masked_cells

        # Filter all per-cell arrays along axis 1
        self.x      = self.x[:, cell_mask]
        self.y      = self.y[:, cell_mask]
        self.h      = self.h[:, cell_mask]
        self.A      = self.A[:, cell_mask]
        self.label  = self.label[:, cell_mask]
        self.aminor = self.aminor[:, cell_mask]
        self.amajor = self.amajor[:, cell_mask]
        self.theta  = self.theta[:, cell_mask]
        self.ecc    = self.ecc[:, cell_mask]

        # dx, dy have same cell axis, recompute or slice
        self.dx = np.ma.diff(self.x, axis=0)
        self.dy = np.ma.diff(self.y, axis=0)

        # density is per frame, recompute from filtered A
        self.density = 10 ** 6 / np.ma.mean(self.A, axis=1)


