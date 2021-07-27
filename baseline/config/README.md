# Baseline for the configuration task

The training scripts and a pre-trained model for the configuration task can be
found here. This model tries to find good parameter settings by using the hyperparameter
tuning tool [SMAC](https://automl.github.io/SMAC3/master/installation.html).
The pre-trained model can be found in `agents/config.py`. To use this baseline
as a starting point for deriving an improved configuration with SMAC, do the following:

1. Make sure SMAC is installed (instructions can be found [here](https://automl.github.io/SMAC3/master/installation.html))
and scipy version 1.6.x or older is used.

2. Modify the file `parameters_to_tune.txt` to include (or exclude) all
parameters you want (or don't want) to tune. A list of all important 
parameters can be found in `scip_parameters.txt`. Run
`python ../../baseline/config/generateParameters.py` from your submission folder.

3. Run `python ../../baseline/config/run_training.py PROBLEM`. The optional arguments
`-t TIMELIMIT` (timelimit for evaluation of each instance in SMAC; default: 300),
`-i NINSTANCES` (number of instances to use for training; default: 10),
`-s SEED` (random seed for selection of training instances) and 
`-e NEVALUATIONS` (number of SMAC evaluations per instance; default: 10) can be set.
If more than 40 different parameters should be solved, SMAC's initial
design needs to be changed (since the default design can only handle <40
dimensions). To change that, add for example `initial_design=RandomConfigurations`
to the `SMAC4HPO` call (and add `from smac.initial_design.random_configuration_design import RandomConfigurations`)
to the file.

4. After SMAC finishes, the best found parameter settings are printed and can be
added to your Policy class.

5. For your submission, you need to include `init.sh` and `conda.yaml` as well as
`agents/config.py` with your (updated) Policy class.
