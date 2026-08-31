from pathlib import Path

def create_run_fname(config):

    N     = config['simulation']['Nvertices']
    seed  = config['simulation']['seed']

    gamma = int(config['physics']['gamma'])
    v0    = int(config['physics']['v0'])
    taup  = int(config['physics']['taup'] * 100)
    eta   = int(config['physics']['eta'] * 100)

    dirname = f"gamma{gamma}_v0{v0}_taup{taup}_eta{eta}"
    fname   = f"N{N}_seed{seed}"

    return dirname, fname
