#!/bin/bash

display_usage() {
	echo -e "Usage: $0 [gpu,nogpu]"
}

# if less than one arguments supplied, display usage
if [  $# -lt 1 ]
then
	display_usage
	exit 1
fi

case $1 in
    gpu)
        export SINGULARITY_NV=1
        ;;
    nogpu)
        export SINGULARITY_NV=0
        ;;
    *)
	display_usage
	exit 1
esac

export SINGULARITY_HOME=`realpath home/$1`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp,$(realpath data/instances):$SINGULARITY_HOME/instances:ro"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NETWORK=none

singularity run base.sif
