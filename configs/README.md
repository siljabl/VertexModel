# CONFIG PARAMETERS
                
# Simulation
seed         : Random number generator seed.
size         : Linear size of the lattice domain.
t1_threshold : Edge-length threshold below which T1 transitions are triggered.
t1_offset    : Length offset after a T1 transition. New edges have length
               t1_threshold + t1_offset.
edge_limit   : Minimum allowed junction length for dividing cells, expressed as
               a fraction of the mean junction length.

# Integration
dt           : Numerical integration time step.
period       : Number of integration steps between saved simulation frames.
n_transient  : Number of initial saved frames excluded as transient dynamics.
n_frames     : Number of saved simulation frames retained after the transient period.

# Forces
gamma        : Surface tension coefficient.
Lambda       : Surface-tension anisotropy parameter.
v0           : Magnitude of self-propulsion velocity.
taup         : Persistence time of self-propulsion direction.
eta          : Drag coefficient for vertex--vertex interactions.

# Cells
distribution : Probability distribution used to initialise cell volumes.
s            : Shape parameter $\sigma$ of the log-normal distribution (`scipy.stats.lognorm`).
scale        : Scale parameter $\exp(\mu)$ of the log-normal distribution (`scipy.stats.lognorm`).
Vstd         : Standard deviation of the initial cell-volume distribution.

# Growth and division
g            : Per-cell volume-growth rate.
k_div        : Division-rate coefficient.
V_div        : Cell volume threshold for division.





## Growth and division

- `g`: Per-cell volume-growth rate.
- `k_div`: Division-rate coefficient.
- `V_div`: Cell-volume threshold for division.
