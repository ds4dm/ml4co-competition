#!/bin/bash

display_usage() {
	echo -e "Usage: $0 team task benchmark"
}

# if less than three arguments supplied, display usage
if [  $# -lt 3 ]
then
	display_usage
	exit 1
fi

export SINGULARITY_HOME=`realpath home/$1`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp,$(realpath instances):$SINGULARITY_HOME/instances:ro"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1
export SINGULARITY_NETWORK=none

singularity exec --net base.sif bash run_evaluation.sh ${@:2}
