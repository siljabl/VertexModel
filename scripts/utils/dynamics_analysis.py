import numpy as np



def compute_msd(positions, max_lag=None, mask_empty=True):
    """
    Compute mean-squared displacement (MSD) as a function of lag time.

    Parameters
    ----------
    positions : array, shape (T, N, d)
        Positions of N cells over T time points in d dimensions.
        positions[t, i, :] is the position of cell i at time t.
    max_lag : int or None
        Maximum time lag (in frames) to compute MSD for.
        If None, use max_lag = T - 1.
    mask_empty : bool

    Returns
    -------
    lag_times : (L,) array
        Lag times in frames (0, 1, ..., max_lag).
    msd : (L,) array
        MSD at each lag time.
    """
    positions = np.asarray(positions)
    T, N, d = positions.shape

    if max_lag is None or max_lag > T - 1:
        max_lag = T - 1

    lag_times = np.arange(max_lag + 1)
    msd = np.zeros_like(lag_times, dtype=float)

    # MSD at lag 0 is zero by definition
    msd[0] = 0.0

    # Loop over lag times
    for tau in range(1, max_lag + 1):

        # displacements: positions[t+tau] - positions[t]
        disp = (positions[tau:T, :, :] - positions[0:T-tau, :, :])
        
        # squared displacement per cell and time
        sq_disp = np.ma.sum(disp**2, axis=2)          # shape (T - tau, N)

        if mask_empty:
            mask = (np.ma.sum(positions[tau:T, :, :], axis=2) > 0) * (np.ma.sum(positions[0:T-tau, :, :], axis=2) > 0)
            sq_disp = sq_disp[mask]          # shape (T - tau, N)

        # average over time and cells
        msd[tau] = sq_disp.mean()

    return lag_times, msd



def build_neighbours_distance(positions_t0, r_cut):
    """
    Build neighbor list by distance at reference time t0.

    Parameters
    ----------
    positions_t0 : (N, d) array
        Positions of N cells at time t0.
    r_cut : float
        Distance cutoff for neighbors.

    Returns
    -------
    neighbors : list of lists
        neighbors[i] is a list of indices of neighbor cells of cell i.
    """

    positions_t0 = np.asarray(positions_t0)
    N, d = positions_t0.shape

    neighbors = [[] for _ in range(N)]

    # Compute pairwise distances (O(N^2), OK for moderate N)
    for i in range(N):
        # Vector from cell i to all cells
        dr = positions_t0 - positions_t0[i]
        # Euclidean distances
        dist = np.linalg.norm(dr, axis=1)
        # Neighbors within r_cut, excluding i itself
        neigh = np.where((dist < r_cut) & (dist > 0.0))[0]
        neighbors[i] = neigh.tolist()

    return neighbors



def compute_cage_displacement(positions_t0, positions_t1, neighbors):
    positions_t0 = np.asarray(positions_t0)
    positions_t1 = np.asarray(positions_t1)
    N, d = positions_t0.shape

    cage_disp = np.zeros_like(positions_t0)

    for i in range(N):
        dr_i = positions_t1[i] - positions_t0[i]  # lab-frame displacement

        neigh = neighbors[i]
        if len(neigh) == 0 or np.sum(positions_t1[i]) == 0:
            # no neighbors found: use lab-frame displacement
            cage_disp[i] = 0 #dr_i
            continue

        R_t0 = positions_t0[neigh].mean(axis=0)  # cage center at t0
        R_t1 = positions_t1[neigh].mean(axis=0)  # cage center at t1

        dR_i = R_t1 - R_t0
        cage_disp[i] = dr_i - dR_i

    return cage_disp



def compute_fractional_cage_change(neighbors_t0, neighbors_t1):
    """
    Compute fractional cage change for each cell between t0 and t1.

    For cell i:
        f_i = 1 - |N_i(t0) ∩ N_i(t1)| / |N_i(t0) ∪ N_i(t1)|

    Parameters
    ----------
    neighbors_t0 : list of lists
        neighbors_t0[i] is the list of neighbor indices of cell i at t0.
    neighbors_t1 : list of lists
        neighbors_t1[i] is the list of neighbor indices of cell i at t1.

    Returns
    -------
    frac_change : (N,) array
        Fractional cage change per cell in [0, 1].
    """
    N = len(neighbors_t0)
    assert len(neighbors_t1) == N

    frac_change = np.zeros(N, dtype=float)

    for i in range(N):
        S0 = set(neighbors_t0[i])
        S1 = set(neighbors_t1[i])

        if not S0 or not S1:
            # No neighbours in either frame: define change as 0
            frac_change[i] = 999.0
            continue

        inter = S0.intersection(S1)
        union = S0.union(S1)

        if len(union) == 0:
            frac_change[i] = 0.0
        else:
            frac_change[i] = 1.0 - len(inter) / len(union)

    return frac_change
