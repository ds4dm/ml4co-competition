#!/bin/bash

display_usage() {
	echo -e "Usage: $0 BENCHMARK"
  echo -e "  BENCHMARK: problem benchmark to evaluate (item_placement, load_balancing, anonymous)"
  echo -e "  OPTIONS:"
    echo -e "    -j (--njobs) : number of parallel jobs for  sample generation"
}

# if less than one arguments supplied, display usage
if [  $# != 1 ]
then
	display_usage
	exit 1
fi

python train_files/01_generate_dataset.py ${@:1}
python train_files/02_train.py $1 -g 0
mkdir agents/trained_models
cp -r train_files/trained_models/$1 agents/trained_models/
