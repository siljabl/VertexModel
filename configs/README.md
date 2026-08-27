# CONFIG PARAMETERS
                
# Lattice
seed  : random number generator seed
Ngrid : number of vertices in each dimension. Ncell = Ngrid**2 / 3
Lgrid : length of lattice
rgrid : length scale of triangular lattice

# Cell
A0    : initial cell area
V0    : cell volume
s     : parameter of scipy.stats.lognorm
scale : parameter of scipy.stats.lognorm

# Forces
gamma  : surface tension
Lambda : ratio of lateral to apical-basal surface tension
tauV   : inverse increase rate in V0 unit
v0     : self-propulsion velocity
taup   : self-propulsion persistence time
eta    : vertex-vertex pair drag coefficient

# Integration
dt      : integration time step
delta   : length below which T1s are triggered
epsilon : edges have length delta+epsilon after T1s
period  : period between frames
Nframes : number of frames in simulation

# Division
Vth_ratio : ration of volume threshold for division to mean volume