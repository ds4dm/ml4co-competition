# Machine Learning for Combinatorial Optimization
## [NeurIPS 2021 Competition](https://www.ecole.ai/2021/ml4co-competition/)

This repository contains the evaluation scripts that we will use to evaluate the competitors, as well as some baseline implementations for each of the three tasks of the competition (primal, dual, config).

Note that we employ a singularity container to evaluate the code of each team, as well as conda and pip to manage the software dependencies of each team. For development purposes or to train your ML models, you do not have to use the singularity container. However, the use of conda and/or pip is strongly encouraged to make the installation process easier on our side.

We encourage participants to test their code within a singularity container using the provided pipeline before they make a submission, so that they can detect and fix potential problem in advance.

## Benchmark files

Participants will find the training and validation instances for
each problem benchmark [here](https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view?usp=sharing).
The `instance` folder is to be placed at the root of this repository.

Note that for each benchmark we provide a pre-defined split of the
instance files into a training (train) and a validation set (valid),
however we do not impose any restriction on how those instances are used.
All the provided instances can be considered training instances.

The test instances, which will be used to evaluate the participants,
will not be made public before the end of the competition.

## Submissions

The main idea of our evaluation pipeline is that we will keep a separate `home/TEAM` folder for each team, where we will place their submission. A team submission then consists in a single folder with the following structure:
 - `conda.yaml` the file that specifies the conda and pip packages to be installed
 - `init.sh` (optional) the initialization script, with additional installation commands if needed
 - `agents/primal.py` the code of the team's agent for the primal task, if they want to compete in the primal task
 - `agents/dual.py` the code of the team's agent for the dual task, if they want to compete in the dual task
 - `agents/config.py` the code of the team's agent for the config task, if they want to compete in the config task
 - `XXX` any other necessary file (for example, an ML model and its parameters)

A minimal example of such files can be found in the `home/naive_baseline` folder.

## Evaluation pipeline, without singularity

The Python scripts required to evaluate a submission, which are common to every
participant, can be found in the `common` folder:
 - `common/environments.py` definition of the POMDP environments for each task
 - `common/rewards.py` definition of the reward functions for each task
 - `common/evaluate.py` evaluation script

The evaluation instances for the three problem benchmarks must be accessible within
the `instances` folder at the root of this repository.

The evaluation for each task and each problem benchmark can be run as follows:
```
cd home/TEAM

python ../../common/evaluate.py primal item_placement
python ../../common/evaluate.py primal load_balancing
python ../../common/evaluate.py primal anonymous

python ../../common/evaluate.py dual item_placement
python ../../common/evaluate.py dual load_balancing
python ../../common/evaluate.py dual anonymous

python ../../common/evaluate.py config item_placement
python ../../common/evaluate.py config load_balancing
python ../../common/evaluate.py config anonymous
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

## Evaluation pipeline, with singularity

To evaluate their submission within singularity, participants must place their code and files within a `home/TEAM` folder, and then go through the following steps.

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
