#!/bin/bash

display_usage() {
	echo -e "Usage: $0 TEAM"
}

# if less than one arguments supplied, display usage
if [  $# -lt 1 ]
then
	display_usage
	exit 1
fi

TEAM_DIR="submissions/$1"

# check if team exists
if [ ! -d $TEAM_DIR ]
then
    echo "Error: directory $TEAM_DIR does not exist."
	exit 1
fi

export SINGULARITY_HOME=`realpath $TEAM_DIR`
export SINGULARITY_BIND="$(mktemp -d):/tmp,$(mktemp -d):/var/tmp"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1

singularity exec singularity/base.sif bash init.sh
