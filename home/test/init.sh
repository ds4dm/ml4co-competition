source /opt/mamba/init.bash
conda env remove -n ml4co
conda env create -n ml4co -f environment.yaml
conda activate ml4co
# installation commands for additional dependencies can go here

conda install -c conda-forge/label/ecole_dev ecole
