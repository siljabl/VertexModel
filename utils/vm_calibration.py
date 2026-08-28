import numpy as np


def cell_density(config):
    """
    Global cell density (cells per unit area) from hexagonal grid setup.
    Output is in units of cells per mm^2

    Ngrid: number of vertices per side in the hexagonal lattice
    Lgrid: system size in the same length units as positions
    """
    Ngrid = config['simulation']['Nvertices']
    Lgrid = config['simulation']['Lgrid']


    Ncell = Ngrid**2 / 3.0
    Agrid = (np.sqrt(3) / 2.0) * Lgrid**2

    rho = Ncell / Agrid

    return int(rho * 10**6)


def cell_density_from_Ngrid(Ngrid, Lgrid=600):
    """
    Global cell density (cells per unit area) from hexagonal grid setup.
    Output is in units of cells per mm^2

    Ngrid: number of vertices per side in the hexagonal lattice
    Lgrid: system size in the same length units as positions
    """

    Ncell = Ngrid**2 / 3.0
    Agrid = (np.sqrt(3) / 2.0) * Lgrid**2

    rho = Ncell / Agrid

    return np.astype(rho * 10**6, int)


def cell_volume_from_density(rho):
    """
    Average cell volume as function of cell number density.
    Based on empirical fit to pixelwise average:
        V0 ≈ 5200 - 1.2*rho + 1.5e-4*rho**2
    """
    return 5200 - 1.2 * rho + 1.5e-4 * rho**2



def cell_volume(config):
    """
    Convenience: average cell volume from (Ngrid, Lgrid) by
    combining cell_density and cell_volume_from_density.
    """
    rho = cell_density(config)
    return cell_volume_from_density(rho)


def cell_division_volume(config):

    Vth_ratio = config["calibration"]["Vth_ratio"]

    V0 = cell_volume(config)

    return Vth_ratio * V0
    
