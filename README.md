# Machine Learning for Combinatorial Optimization
## NeurIPS 2021 Competition

This repository contains the evaluation scripts which will be used to evaluate the competitors, as well as some baseline implementations for each of the three tasks of the competition (primal, dual, config). We employ a singularity container to evaluate the code of each team, as well as conda and pip to manage the software dependencies of each team.

For development purposes or to train your ML models, you do not have to use the singularity container. However, the use of conda and/or pip is strongly encouraged to make the installation process easier on our side.

Still, we encourage participants to test their code within our singularity pipeline before they make a submission, so that they can detect and fix potential problem in advance.

In addition to the scripts in this repository, participants can download the training and validation instances for each problem benchmark [here]().Note that for each benchmark we provide a pre-defined split of the benchmarks into a training and a validation set, however we do not impose any restriction on how the instances are used. All the provided instances can be considered traiing instances.

## Submissions

The main idea is that we will keep a separate `home/TEAM` folder for each team, where we will place their submissions. A team submission then consists in a single folder with the following structure:
 - `conda.yaml` the file that specifies the conda and pip packages to be installed
 - `init.sh` (optional) the initialization script, with additional installation commands if needed
 - `tasks/agents/primal.py` the code of the team's agent for the primal task, if they want to compete in the primal task
 - `tasks/agents/dual.py` the code of the team's agent for the dual task, if they want to compete in the dual task
 - `tasks/agents/config.py` the code of the team's agent for the config task, if they want to compete in the config task
 - `tasks/agents/XXX` any other necessary file (for exaqmple, an ML model and its parameters)

A minimal example of such files can be found in the `home/naive_baseline` folder.

## Evaluation pipeline without singularity

Additional Python files are required to evaluate a submission, which can also be fond in the `home/naive_baseline` folder:
 - `tasks/environments.py` definition of the POMDP environments for each task
 - `tasks/rewards.py` definition of the reward functions for each task
 - `tasks/evaluate.py` evaluation script

The evaluation instances for the three problem benchmarks must be accessible through the `home/TEAM/instances` folder (either by directly placing them here, or by using a synlink). Then, the evaluation for each task and each problem benchmark can be run as follows:
```
cd home/TEAM

python tasks/evaluate.py primal item_placement
python tasks/evaluate.py primal load_balancing
python tasks/evaluate.py primal anonymous

python tasks/evaluate.py dual item_placement
python tasks/evaluate.py dual load_balancing
python tasks/evaluate.py dual anonymous

python tasks/evaluate.py config item_placement
python tasks/evaluate.py config load_balancing
python tasks/evaluate.py config anonymous
```

Those scripts will output results in the following files:
```
home/TEAM/results/primal/1_item_placement.csv
home/TEAM/results/primal/2_load_balancing.csv
home/TEAM/results/primal/3_anonymous.csv

home/TEAM/results/dual/1_item_placement.csv
home/TEAM/results/dual/2_load_balancing.csv
home/TEAM/results/dual/3_anonymous.csv

home/TEAM/results/config/1_item_placement.csv
home/TEAM/results/config/2_load_balancing.csv
home/TEAM/results/config/3_anonymous.csv
```

## Evaluation pipeline with singularity

To evaluate their subnission within singularity, participants can place their code within a `home/TEAM` folder, place the instances files within an `instances` folder, and then go through the following steps.

Note: ideally you can place the instances in a single location, and use a synlink as follows:
```bash
ln -s /path/to/instances instances
```

### Singularity set-up (only once)

Make sure `singularity` is installed and available.

Build the Singularity image. The script is configured to do the build remotely (--remote), which requires to create a [Sylab account](https://cloud.sylabs.io/home).
```bash
sh 01_singularity_build.sh
```

### Team set up (only once per team)

Set up an `ml4co` conda environment within the team's container, based on the team's `conda.yaml` and `init.sh` files. Requires internet access to download dependencies.
```bash
sh 02_participant_init.sh TEAM
```

### Team evaluation

Run the evaluation script within the team's container. No internet access.
```bash
sh 03_participant_run.sh TEAM primal item_placement
sh 03_participant_run.sh TEAM primal load_balancing
sh 03_participant_run.sh TEAM primal anonymous

sh 03_participant_run.sh TEAM dual item_placement
sh 03_participant_run.sh TEAM dual load_balancing
sh 03_participant_run.sh TEAM dual anonymous

sh 03_participant_run.sh TEAM config item_placement
sh 03_participant_run.sh TEAM config load_balancing
sh 03_participant_run.sh TEAM config anonymous
```
