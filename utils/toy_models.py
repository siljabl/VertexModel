import numpy as np

def run_division_no_history(initial_volumes, Ntime,
                            k_div, V_div, g, 
                            dt=0.0832, Nmax=1e6):

    # Initial number of cells. Same as cell number target
    Ntarget = len(initial_volumes)
    Ncells  = Ntarget

    # Initialise volume array
    volumes_prev = np.copy(initial_volumes)

    Ncount  = [Ncells]
    Vcount  = [np.mean(initial_volumes)]

    for t in range(1, Ntime):

        # Compute removal probability
        p_dead = np.clip(Ncells / Ntarget - 1, 0.0, 1.0)

        # Pick cells to remove. Dying cells are picked randomly.
        r_dead = np.random.rand(Ncells)
        dead   = (r_dead < p_dead)

        # Upadte cell volumes and number of cells
        volumes_prev = volumes_prev[~dead]
        Ncells = len(volumes_prev)

        # Linear cell growth
        volumes_before_div = volumes_prev + g*dt

        # Compute division probability
        rate_div  = k_div * (volumes_before_div - V_div)
        rate_div[rate_div < 0] = 0
        p_div = 1 - np.exp(-rate_div * dt)

        # Pick cells for division
        r_div  = np.random.rand(Ncells)
        divide = (r_div < p_div)

        # Number of dividing cells
        Ndiv = np.sum(divide)

        # upadte division volumes
        mother_indices = np.where(divide)[0]

        volumes_after_div = np.zeros(Ncells + Ndiv)
        volumes_after_div[:Ncells] = volumes_before_div

        volumes_after_div[mother_indices] *= 0.5
        volumes_after_div[Ncells:] = volumes_after_div[mother_indices]

        # Update cell counter
        Ncells  = len(volumes_after_div)
        Ncount.append(Ncells)
        Vcount.append(np.mean(volumes_after_div))

        volumes_prev = volumes_after_div

        if Ncells >= Nmax:
            break

    return volumes_prev, Ncount, Vcount



# # Algorithm for simulating division rule
# # Takes an array of initially uniformly distributed cell volume and let them grow linearly with growth rate tauV
# # Picks volumes to divide stochastically, with probability p = a(V_i / V_0 - b)
# def run_division_with_history(initial_volumes, Ntime, 
#                               k_div, V_div, g,
#                               dt=0.0832, max_factor=10):

#     # Initial number of cells
#     Ncells = len(initial_volumes)

#     # Initialise volume array
#     Nmax    = int(max_factor * Ncells)
#     volumes = np.ma.array(np.zeros([Ntime, Nmax]))
#     volumes[0, :Ncells] = initial_volume_distribution
#     volumes.mask        = volumes == 0


#     for t in range(1, Ntime):

#         # Linear cell growth
#         volumes[t] = volumes[t-1] + g*dt

#         # Compute division probability
#         rate_div  = k_div * (volumes[t] - V_div)
#         rate_div[rate_div < 0] = 0
#         p_div = 1 - np.exp(-rate_div * dt)

#         # Pick cells for division
#         r_div  = np.random.rand(Nmax)
#         divide = (r_div < p_div)

#         # Number of dividing cells
#         Ndiv = np.sum(divide)

#         # Split dividing cell volumes by half
#         volumes[t][divide] *= 0.5

#         if Ncells + Ndiv >= Nmax:
#             print("Max cell number exceeded at time ", t)
#             break

#         # Copy volume of divided mother cell to daughter cell
#         volumes[t, Ncells:Ncells + Ndiv] = volumes[t][divide].compressed()

#         # Update cell counter
#         Ncells += Ndiv

#     return volumes, t