#!/bin/bash

display_usage() {
	echo -e "Usage: $0 team"
}

# if less than two arguments supplied, display usage
if [  $# -lt 1 ]
then
	display_usage
	exit 1
fi

export SINGULARITY_HOME=`realpath home/$1`:/home/user
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp,$(realpath data/instances):/home/user/instances:ro"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1
export SINGULARITY_NETWORK=none

singularity run --net base.sif
