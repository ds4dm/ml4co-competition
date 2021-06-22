#!/bin/bash

display_usage() {
	echo -e "Usage: $0 team"
}

# if less than one arguments supplied, display usage
if [  $# -lt 1 ]
then
	display_usage
	exit 1
fi

export SINGULARITY_HOME=`realpath home/$1`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1

singularity exec base.sif bash init.sh
