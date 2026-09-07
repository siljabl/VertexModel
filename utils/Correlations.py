import os
import pickle
import numpy as np
import correlation_core as compute

from pathlib import Path
from datetime import datetime
from paths_config import SIM_RAW_DIR, SIM_PROC_DIR
from matrix_preprocessing import detrend_entire_matrix


class AutocorrBase:
    """
    Base class that holds temporal and spatial correlations and provides
    generic compute / save / load logic.
    """

    def __init__(self):
        # Core correlation storage
        self.temporal      = {}
        self.temporal_cell = {}   # per-cell temporal correlations
        self.spatial       = {}
        self.t_array       = {}
        self.r_array       = {}
        self.density       = np.array([])

        # Simple log of computation times for each correlation
        self.log = {
            't':      {},
            't_cell': {},
            'r':      {},
        }

    # ---------- common compute methods ----------

    def compute_spatial(self, positions_x, positions_y, variable,
                        variable_name, dr, r_max, t_avrg=False,
                        overwrite=False,
                        add_lag0=True):
        """
        Generic spatial autocorrelation.

        positions_x, positions_y : arrays with shape (Nframes, Npoints)
        variable                 : scalar or vector field (masked arrays)
        add_lag0                 : if True, prepend lag-0 point C(0)=1 at r=0.
        """

        if not overwrite and variable_name in self.spatial:
            print(f"Spatial autocorrelation of {variable_name} already exists.")
            return

        # Compute spatial correlation
        Cr = compute.general_spatial_correlation(
            positions_x, positions_y, variable,
            dr=dr, r_max=r_max, t_avrg=t_avrg
        )

        C_r   = Cr['C_norm']
        r_arr = Cr['r_bin_centers']

        # Prepend C(0)=1 and r=0
        if add_lag0:
            C0   = np.ma.ones((len(C_r), 1))  # data=1, mask=False
            C_r  = np.ma.concatenate([C0, C_r], axis=1)
            r_arr = np.concatenate([[0], r_arr])

        self.spatial[variable_name]  = C_r
        self.r_array[variable_name]  = r_arr
        self.log['r'][variable_name] = datetime.today().strftime('%Y/%m/%d_%H:%M')

        return Cr



    def compute_temporal(self, variable, variable_name,
                         t_max, t_avrg=False, overwrite=False,
                         df=1.0, mean_var='r'):
        """
        Generic temporal autocorrelation.

        df       : frame-to-time factor (simulation uses this for hours).
        mean_var : 'r'  -> store in self.temporal
                  'cell' -> store in self.temporal_cell
        """

        assert mean_var in ('r', 'cell')

        store_dict = self.temporal if mean_var == 'r' else self.temporal_cell

        if not overwrite and variable_name in store_dict:
            print(f"Temporal autocorrelation of {variable_name} already exists in {mean_var}.")
            return

        # Compute temporal correlation
        Ct = compute.general_temporal_correlation(variable, t_max=t_max, t_avrg=t_avrg)

        store_dict[variable_name]    = Ct['C_norm']
        self.t_array[variable_name]  = np.arange(t_max) * df

        if mean_var == 'r':
            self.log['t'][variable_name]      = datetime.today().strftime('%Y/%m/%d_%H:%M')
        else:
            self.log['t_cell'][variable_name] = datetime.today().strftime('%Y/%m/%d_%H:%M')

        return Ct

    # ---------- save / load helpers ----------

    def save_state(self, path):
        """
        Save correlations and metadata to a pickle file.
        """

        state = {
            'temporal':      self.temporal,
            'temporal_cell': self.temporal_cell,
            'spatial':       self.spatial,
            't_array':       self.t_array,
            'r_array':       self.r_array,
            'density':       self.density,
            'log':           self.log,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"State saved to {path}")


    def load_state(self, path):
        """
        Load correlations and metadata from a pickle file.
        """

        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.temporal      = state.get('temporal', {})
        self.temporal_cell = state.get('temporal_cell', {})
        self.spatial       = state.get('spatial', {})
        self.t_array       = state.get('t_array', {})
        self.r_array       = state.get('r_array', {})
        self.density       = state.get('density', np.array([]))
        self.log           = state.get('log', self.log)
        print(f"State loaded from {path}")



class ExperimentalAutocorrelations(AutocorrBase):
    """
    Correlation container for experimental data.

    Parameters
    ----------
    path : str
        Path to the .autocorr file to load / save.
    """

    def __init__(self, path):
        super().__init__()
        self.path = path

        if os.path.isfile(self.path):
            self.load_state(self.path)
        else:
            print(f"No saved state at {self.path}. Starting fresh.")


    def save(self):
        self.save_state(self.path)


    def load(self):
        self.load_state(self.path)


    # Override compute_spatial to pass positions in (x,y) form and keep lag-0
    def compute_spatial(self, positions, variable, variable_name,
                        dr, r_max, t_avrg=False, overwrite=False):
        # positions is (2, Nframes, Npoints) as in your exp code
        fmax = variable.shape[-2]
        x = positions[0][:fmax]
        y = positions[1][:fmax]

        return super().compute_spatial(
            x, y, variable, variable_name,
            dr, r_max, t_avrg=t_avrg, overwrite=overwrite,
            add_lag0=True
        )

    # mean_var='r'/'cell' uses base method
    def compute_temporal(self, variable, variable_name, t_max,
                         df=1.0, mean_var='r', t_avrg=False, overwrite=False):
        return super().compute_temporal(
            variable, variable_name, t_max,
            t_avrg=t_avrg, overwrite=overwrite,
            df=df, mean_var=mean_var
        )




class SimulationAutocorrelations(AutocorrBase):
    """
    Correlation container for simulation data.

    Handles mapping between raw simulation pickle files in SIM_RAW_DIR and
    processed correlation files in SIM_PROC_DIR.
    """
    
    def __init__(self, in_path=None, out_path=None, path_addition=''):
        super().__init__()
        assert in_path is not None or out_path is not None

        if in_path is not None:
            in_path = Path(in_path)
            try:
                rel = in_path.relative_to(SIM_RAW_DIR)
            except ValueError:
                rel = in_path.name
            filename = str(Path(rel).with_suffix(""))
        else:
            out_path = Path(out_path)
            try:
                rel = out_path.relative_to(SIM_PROC_DIR)
            except ValueError:
                rel = out_path.name
            filename = str(Path(rel).with_suffix(""))

        self.in_path       = f"{SIM_RAW_DIR}/{filename}.p"
        self.out_path      = f"{SIM_PROC_DIR}/{filename}.autocorr"
        self.path_addition = path_addition

        full_out = f"{path_addition}{self.out_path}"
        if os.path.exists(full_out):
            self.load_state(full_out)
        else:
            print(f"No saved state file found at {full_out}. Starting fresh.")


    def save(self):
        self.save_state(f"{self.path_addition}{self.out_path}")


    # Spatial: positions shape (Nframes, Ncells, 2), include lag-0 point
    def compute_spatial(self, positions, variable, variable_name,
                        dr, r_max, t_avrg=False, overwrite=False):
        x = positions[:, :, 0]
        y = positions[:, :, 1]
        return super().compute_spatial(
            x, y, variable, variable_name,
            dr, r_max, t_avrg=t_avrg, overwrite=overwrite,
            add_lag0=True
        )

    # Temporal: same API as experimental, but with df for time units
    def compute_temporal(self, variable, variable_name, t_max,
                         df=1.0, mean_var='r', t_avrg=False, overwrite=False):
        return super().compute_temporal(
            variable, variable_name, t_max,
            t_avrg=t_avrg, overwrite=overwrite,
            df=df, mean_var=mean_var
        )



def compute_cell_correlation(autocorr_obj, parameter_map, args):

    # -------- Spatial correlations --------
    if args.var == "r" or args.var == "all":
        rmax = int(550 * args.rfrac)

        for name, (pos, var) in parameter_map.items():

            # Compute spatial correlation
            if args.param in (name, 'all'):
                autocorr_obj.compute_spatial(positions=pos, 
                                             variable=var, 
                                             variable_name=name, 
                                             dr=args.dr, 
                                             r_max=rmax, 
                                             t_avrg=args.t_avrg, 
                                             overwrite=args.overwrite)


    # -------- Temporal correlations --------
    if args.var == 't' or args.var == 'all':
        tmax = int(len(parameter_map['hh'][1]) * args.tfrac)

        for name, (pos, var) in parameter_map.items():
            if args.param in (name, 'all'):

                # Detrend if taking correlation w.r.t. cell mean
                if args.mean_var == "cell":
                    var = detrend_entire_matrix(var)

                # Compute temporal correlation
                autocorr_obj.compute_temporal(variable=var,
                                              variable_name=name,
                                              t_max=tmax,
                                              df=args.dt,
                                              mean_var=args.mean_var,
                                              t_avrg=args.t_avrg,
                                              overwrite=args.overwrite)

    # Save autocorrelations
    autocorr_obj.save()