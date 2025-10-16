# #!/bin/bash

# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup100/ --dr=2
# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup150/ --dr=2
# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup200/ --dr=2
# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup250/ --dr=2
# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup300/ --dr=2
# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup350/ --dr=2
# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup400/ --dr=2
# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup450/ --dr=2
# python analysis/compute_correlations.py ../../../../hdd_data/silja/VertexModel_data/simulated/raw/nodivision_20250917_N30_L64_Lambda100_v0100_taup500/ --dr=2


#!/bin/bash

# Iterate through folders matching the pattern *date*/
for folder in data/simulated/raw/nodivision_20250919*; do
    folder_name=${folder}/
    
    # Execute the Python script with the folder name as argument
    echo "Processing folder: $folder_name"
    python analysis/compute_ensemble_average.py "$folder_name" --compute_correlations --mean_var=cell
done
