# Configuration parameters

## Simulation

- `seed`: Seed for the random-number generator.
- `n_vertices`: Number of vertices in each dimension of the initial lattice.
- `size`: Linear size of the lattice domain.
- `t1_threshold`: Edge-length threshold below which a T1 transition is
  triggered.
- `t1_offset`: Length offset applied after a T1 transition. The newly created
  edge has length `t1_threshold + t1_offset`.
- `edge_limit`: Minimum permitted junction length for cell division, expressed
  as a fraction of the mean junction length. Cells with shorter junctions do
  not divide.

## Integration

- `dt`: Numerical integration time step.
- `period`: Number of integration steps between saved simulation frames.
- `n_transient`: Number of initial saved frames treated as transient dynamics
  and excluded from analysis.
- `n_frames`: Number of saved frames retained after the transient phase.

## Forces

- `gamma`: Surface-tension coefficient.
- `Lambda`: Surface-tension anisotropy parameter.
- `v0`: Magnitude of the self-propulsion velocity.
- `taup`: Persistence time of the self-propulsion direction.
- `eta`: Drag coefficient for vertex--vertex interactions.

## Cells

- `distribution`: Probability distribution used to initialise cell volumes.
- `s`: Shape parameter $\sigma$ of the log-normal distribution
  (`scipy.stats.lognorm`).
- `scale`: Scale parameter $\exp(\mu)$ of the log-normal distribution
  (`scipy.stats.lognorm`).
- `Vstd`: Standard deviation of the initial cell-volume distribution.

## Growth and division

- `g`: Per-cell volume-growth rate.
- `k_div`: Division-rate coefficient.
- `V_div`: Cell-volume threshold for division.
