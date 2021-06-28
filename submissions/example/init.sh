# remove any previous environment
conda env remove -n ml4co

# create the environment from the dependency file
conda env create -n ml4co -f conda.yaml
conda activate ml4co

# TODO: remove when 7.0 has been merged into mainstream ecole
conda uninstall ecole -y
conda install -c conda-forge/label/ecole_dev ecole -y

# additional installation commands go here
