# Install Mamba (Conda alternative) through Mambaforge
readonly mamba_installer="Mambaforge-$(uname)-$(uname -m).sh"
readonly mamba_version="4.10.1-4"
readonly mamba_prefix="/opt/mamba"
wget "https://github.com/conda-forge/miniforge/releases/download/${mamba_version}/${mamba_installer}"
bash "${mamba_installer}" -b -p "${mamba_prefix}"
rm "${mamba_installer}"

# Put the Conda initialization script in a file for lazy loading/
# Singularity does all the environment sourcing as shell (only latter calls bash),
# which conda does not support.
# We put the content in a file, manually call bash, and source it.
{
    echo 'eval "$('/opt/mamba/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"'
    echo 'if [ $? -eq 0 ]; then'
    echo '  eval "$__conda_setup"'
    echo 'else' >> ${mamba_prefix}/init.bash
    echo '  if [ -f "/opt/mamba/etc/profile.d/conda.sh" ]; then'
    echo '    . "/opt/mamba/etc/profile.d/conda.sh"'
    echo '  else'
    echo '    export PATH="/opt/mamba/bin:$PATH"'
    echo '  fi'
    echo 'fi'
    echo 'unset __conda_setup'
} >> ${mamba_prefix}/init.bash
