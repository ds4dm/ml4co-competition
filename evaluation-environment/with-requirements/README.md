# Singularity file with user defined requirements

## Building the image
```bash
sudo singularity build base.sif base.def
```

## Running inside the container (evaluators)
```bash
export ML4CO_ENV_PATH="${PWD}/env"
export ML4CO_PKGS_PATH="${PWD}/pkgs"
export ML4CO_TMP_PATH="$(mktemp -d)"

mkdir -p "${ML4CO_ENV_PATH}"
mkdir -p "${ML4CO_PKGS_PATH}"

# Enable GPU support
export SINGULARITY_NV=1
# Hide all files system from local machine
export SINGULARITY_CONTAINALL=1
# Bind local paths for Conda environment
export SINGULARITYENV_CONDA_ENVS_PATH="/ml4co/envs"
export SINGULARITYENV_CONDA_PKGS_DIRS="/ml4co/pkgs"
export SINGULARITY_BIND="${ML4CO_ENV_PATH}:/ml4co/envs,${ML4CO_PKGS_PATH}:/ml4co/pkgs"
# Bind some temporary directory and  the current directory
export SINGULARITY_BIND="${SINGULARITY_BIND},${ML4CO_TMP_PATH}:/tmp,${PWD}:/ml4co/workdir"
# Set current directory as starting directory
export SINGULARITY_TARGET_PWD="/ml4co/workdir"

singuarity shell base.sif
```
