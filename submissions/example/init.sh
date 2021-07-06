# remove any previous environment
conda env remove -n ml4co

# create the environment from the dependency file
conda env create -n ml4co -f conda.yaml

conda activate ml4co

# additional installation commands go here

conda deactivate
