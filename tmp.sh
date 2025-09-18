#!/bin/bash

#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v050_taup100/ -p=hh -v=r -o
#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup100/ -p=hh -v=r -o
#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0150_taup100/ -p=hh -v=r -o
#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0200_taup100/ -p=hh -v=r -o
#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0250_taup100/ -p=hh -v=r -o
#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250918_N30_L64_Lambda100_v0300_taup100/ -p=hh -v=r -o
#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250918_N30_L64_Lambda100_v0350_taup100/ -p=hh -v=r -o
#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250918_N30_L64_Lambda100_v0400_taup100/ -p=hh -v=r -o
#python analysis/compute_correlations.py data/simulated/raw/nodivision_20250918_N30_L64_Lambda100_v0450_taup100/ -p=hh -v=r -o

python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v050_taup100/ -p=hh -v=t -o --mean_var=cell
python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup100/ -p=hh -v=t -o --mean_var=cell
python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0150_taup100/ -p=hh -v=t -o --mean_var=cell
python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0200_taup100/ -p=hh -v=t -o --mean_var=cell
python analysis/compute_correlations.py data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0250_taup100/ -p=hh -v=t -o --mean_var=cell
python analysis/compute_correlations.py data/simulated/raw/nodivision_20250918_N30_L64_Lambda100_v0300_taup100/ -p=hh -v=t -o --mean_var=cell
python analysis/compute_correlations.py data/simulated/raw/nodivision_20250918_N30_L64_Lambda100_v0350_taup100/ -p=hh -v=t -o --mean_var=cell
python analysis/compute_correlations.py data/simulated/raw/nodivision_20250918_N30_L64_Lambda100_v0400_taup100/ -p=hh -v=t -o --mean_var=cell
python analysis/compute_correlations.py data/simulated/raw/nodivision_20250918_N30_L64_Lambda100_v0450_taup100/ -p=hh -v=t -o --mean_var=cell
