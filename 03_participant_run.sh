#!/bin/bash

display_usage() {
	echo -e "Usage: $0 TEAM TASK BENCHMARK [OPTIONS]"
	echo -e "  TEAM: team to evaluate"
    echo -e "  TASK: task to evaluate (primal, dual, config)"
    echo -e "  BENCHMARK: problem benchmark to evaluate (item_placement, load_balancing, anonymous)"
    echo -e "  OPTIONS:"
    echo -e "    -t (--timelimit) T: time limit to process each instance, in seconds"
    echo -e "    -d (--debug): print debug traces"
    echo -e "    -f (--folder): folder of instances to evaluate (train, valid, test)"
}

# if less than three arguments supplied, display usage
if [  $# -lt 3 ]
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
export SINGULARITY_BIND="${SINGULARITY_BIND},$(realpath instances):$SINGULARITY_HOME/../../instances:ro"
export SINGULARITY_BIND="${SINGULARITY_BIND},$(realpath common):$SINGULARITY_HOME/../../common:ro"
export SINGULARITY_CLEANENV=1
export SINGULARITY_CONTAINALL=1
export SINGULARITY_NV=1
export SINGULARITY_NETWORK=none

singularity exec --net base.sif bash ../../common/run_evaluation.sh ${@:2}
