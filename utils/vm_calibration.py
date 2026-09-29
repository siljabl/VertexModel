import numpy as np


def cell_density_from_Ncell(Ncell, Lgrid=600):
    """
    Global cell density (cells per unit area) from hexagonal grid setup.
    Output is in units of cells per mm^2

    Ncell: number of cells
    Lgrid: system size in the same length units as positions
    """

    # Total area of hexagonal lattice
    Agrid = (np.sqrt(3) / 2.0) * Lgrid**2

    # Cells per unit area
    rho = Ncell / Agrid

    # Convert to cells per mm^2 (assuming Lgrid is in µm)
    return np.astype(rho * 10**6, int)


def cell_density_from_Ngrid(Ngrid, Lgrid=600):
    """
    Global cell density (cells per unit area) from hexagonal grid setup.
    Output is in units of cells per mm^2

    Ngrid: number of vertices per side in the hexagonal lattice
    Lgrid: system size in the same length units as positions
    """

    # Number of cells in lattice
    Ncell = Ngrid**2 / 3.0

    # Total area of hexagonal lattice
    Agrid = (np.sqrt(3) / 2.0) * Lgrid**2

    # Cells per unit area
    rho = Ncell / Agrid

    # Convert to cells per mm^2 (assuming Lgrid is in µm)
    return np.astype(rho * 10**6, int)



def cell_density(config):
    """
    Convenience: global cell density (cells per unit area) from config.
    """
    Ngrid = config['simulation']['Nvertices']           # Number of vertices in each dimension
    Lgrid = config['simulation']['Lgrid']               # Length of lattice in µm

    return cell_density_from_Ngrid(Ngrid, Lgrid)


def cell_diameter_from_density(rho):
    """
    Computing average cell diameter from cell density
    """

    A = 10**6/rho
    d = 2 * np.sqrt(A / np.pi)
    return d


def cell_volume_from_density(rho):
    """
    Average cell volume as function of cell number density.
    Based on empirical fit to pixelwise average:
        V0 ≈ 5200 - 1.2*rho + 1.5e-4*rho**2
    """
    return 5200 - 1.2 * rho + 1.5e-4 * rho**2



def cell_volume(config):
    """
    Convenience: average cell volume from config by
    combining cell_density and cell_volume_from_density.
    """
    rho = cell_density(config)

    return cell_volume_from_density(rho)



def division_probability(config, V):

    k_div = config["calibration"]["k_div"]
    V_div = config["calibration"]["V_div"]
    dt    = config["simulation"]["dt"]
    T     = config["simulation"]["period"]

    rate = k_div * (V - V_div)
    if rate < 0:
        rate = 0

    p_div = 1 - np.exp(-rate * dt)

    return p_div



def death_probability(config, A):

    k_death = config["calibration"]["k_death"]
    A_death = config["calibration"]["A_death"]
    dt      = config["simulation"]["dt"]
    T       = config["simulation"]["period"]

    rate = k_death * (A_death - A)
    if rate < 0:
        rate = 0

    p_death = 1 - np.exp(-rate * dt)

    return p_death



def cell_death_area(config, A0=None):
    """
    Threshold area for cell death
    """
    Ath_ratio = config["calibration"]["Ath_ratio"]

    if not A0:
        Lgrid     = config['simulation']['Lgrid']               # Length of lattice in µm
        Ngrid = config['simulation']['Nvertices']           # Number of vertices in each dimension
        
        # Number of cells in lattice
        Ncells = Ngrid**2 / 3.0

        # Total area of hexagonal lattice
        Agrid = (np.sqrt(3) / 2.0) * Lgrid**2

        # Initial average cell area
        A0 = Agrid / Ncells
    
    return Ath_ratio * A0
    
