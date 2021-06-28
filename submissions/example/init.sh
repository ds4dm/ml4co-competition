source /opt/mamba/init.bash
conda env remove -n ml4co
conda env create -n ml4co -f conda.yaml
conda activate ml4co
# installation commands for additional dependencies can go here

# TODO: remove when 7.0 has been merged in maistream ecole
conda uninstall ecole -y
conda install -c conda-forge/label/ecole_dev ecole -y
