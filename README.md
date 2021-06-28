# Machine Learning for Combinatorial Optimization

[NeurIPS 2021 Competition](https://www.ecole.ai/2021/ml4co-competition/)

This repository contains the base code that supports the competitiom, as well as
some code examples and baseline implementations for each of the three tasks of
the competition (primal, dual, config). The structure is as follows:
```
instances/    -> the datasets
common/       -> the common code base of the competition, i.e., the environment, reward, and evaluation scripts
submissions/  -> the team submissions
  example/    -> an example submission
singularity/  -> the singularity image and scripts of our evaluation pipeline
```

The competition's evaluation pipeline requires Python 3.7, as well as `conda`
to install the software dependencies.

## 1. Getting started

### 1.1. Datasets

The training datasets can be downloaded from
[here](https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view?usp=sharing),
and are to be placed inside the `instances` folder. The folder layout looks as follows:
```
instances/
  1_item_placement/
    train/
    valid/
  2_load_balancing/
    train/
    valid/
  3_anonymous/
    train/
    valid/
```

Note that for each benchmark we provide a pre-defined split of the
instance files into a training set (`train`) and a validation set (`valid`).
Participants do not have to respect this arbitrary choice, and
are free to use all the instances in whichever way they like for training,
without any restriction. All the instances included in `train` and `valid`
can be considered training instances.

The test instances used to evaluate the submissions
will not be made public before the end of the competition.

### 1.2. Your submission

The main idea of our evaluation pipeline is that each team's submission
consists of a single folder, i.e., `submissions/YOUR_TEAM_NAME`, which respects the
following structure:
```
submissions/YOUR_TEAM_NAME/
  conda.yaml     -> the conda dependencies file, which specifies the packages to be installed (conda and/or pip)
  init.sh        -> the initialization script that sets up the environment (with additional dependencies not available on conda/pip)
  agents/
    primal.py    -> the code of the team's agent for the primal task, if any
    dual.py      -> the code of the team's agent for the dual task, if any
    config.py    -> the code of the team's agent for the config task, if any
  xxx            -> any other necessary file (for example, the parameters of an ML model)
```

To get started, copy-paste the content of the `submissions/example` folder
into a new folder `submissions/YOUR_TEAM_NAME`, and edit
the file `agents/primal.py`, `agents/dual.py` or `agents/config.py`.

---
**Note**: during the evaluation the `submissions/YOUR_TEAM_NAME` folder will be the working
directory, so that a file `submissions/YOUR_TEAM_NAME/xxx` can be directly accessed via
```Python
with open("xxx") as f:
  do_something
```

---

## 2. Running the evaluation

The Python scripts which are used to evaluate a submission are common to every
participant, and can be found in the `common` folder
```
common/
  environments.py -> definition of the POMDP environments for each task
  rewards.py      -> definition of the reward functions for each task
  evaluate.py     -> evaluation script
```

### 2.1. On the host

#### Environment setup

Your team's environment should be set-up as follows
```bash
cd submissions/YOUR_TEAM_NAME
sh init.sh
```

This should install a conda environment named `ml4co` containing
all the required dependencies.

#### Running the evaluation

Your team's submission can be evaluated on the `valid` instances of
each problem benchmark as follows.

```bash
cd submissions/YOUR_TEAM_NAME
conda activate ml4co

# Primal task
python ../../common/evaluate.py primal item_placement
python ../../common/evaluate.py primal load_balancing
python ../../common/evaluate.py primal anonymous

# Dual task
python ../../common/evaluate.py dual item_placement
python ../../common/evaluate.py dual load_balancing
python ../../common/evaluate.py dual anonymous

# Config task
python ../../common/evaluate.py config item_placement
python ../../common/evaluate.py config load_balancing
python ../../common/evaluate.py config anonymous
```

The result of each team's evaluation is saved here:
```
submissions/YOUR_TEAM_NAME/
  results/
    primal/
      1_item_placement.csv
      2_load_balancing.csv
      3_anonymous.csv
    dual/
      1_item_placement.csv
      2_load_balancing.csv
      3_anonymous.csv
    config/
      1_item_placement.csv
      2_load_balancing.csv
      3_anonymous.csv
```

---
**Note**: you can append `--timelimit T` to the evaluation commands to override
the default time limit for evaluating each instance. For example, setting the
time limit to `T=10` seconds can be usefull for debugging. You can also append
`--debug` to print additional information during the evaluation.

---

### 2.2. Within a singularity container

Team submissions will be evaluated within an Ubuntu-based singularity container,
in order to isolate any side-effect from the code's execution on the host. As such, we provide
the exact singularity image and scripts that we use for evaluation, and we encourage
participants to test their code (installation + evaluation) within the same
container before they make a submission. If your submission will not execute properly
within this container on your side, there is little chance it will on ours.

#### Singularity image set-up (only once)

Make sure `singularity` is installed and available. Then, build the Singularity image.
```bash
sh singularity/01_singularity_build.sh
```

This script is configured to do the build remotely (`--remote`), which requires to create
a [Sylab account](https://cloud.sylabs.io/home). Alternatively, you can also remove the
`--remote` option and do the build locally if you want.

#### Team environment set up (only once per team)

Set up an `ml4co` conda environment of your team within the container.
```bash
sh singularity/02_participant_init.sh YOUR_TEAM_NAME
```

#### Team evaluation

Run the evaluation of your team within the container.

```bash
# Primal task
sh singularity/03_participant_run.sh YOUR_TEAM_NAME primal item_placement
sh singularity/03_participant_run.sh YOUR_TEAM_NAME primal load_balancing
sh singularity/03_participant_run.sh YOUR_TEAM_NAME primal anonymous

# Dual task
sh singularity/03_participant_run.sh YOUR_TEAM_NAME dual item_placement
sh singularity/03_participant_run.sh YOUR_TEAM_NAME dual load_balancing
sh singularity/03_participant_run.sh YOUR_TEAM_NAME dual anonymous

# Config task
sh singularity/03_participant_run.sh YOUR_TEAM_NAME config item_placement
sh singularity/03_participant_run.sh YOUR_TEAM_NAME config load_balancing
sh singularity/03_participant_run.sh YOUR_TEAM_NAME config anonymous
```

---
NOTE

Additional argument such as `--timelimit T` or `--debug` can also be provided here,
and will be passed to the Python evaluation script.

---

## 3. Additional remarks

We will not run the training of your ML models. Please send us
only your final, pre-trained model, ready for evaluation.

We provide an official support to participants via the [Github discussions](https://github.com/ds4dm/ml4co-competition/discussions)
feature. Please direct any technical or general question
regarding the competition there, and feel free to answer
the questions of other participants as well. We will not provide a
privileged support to any of the participants, except in situations where
it concerns a detail about their submission which they do not want to share.

To help us set up your `conda` environment on our side, it can be
useful to include the output of `conda env export --no-builds`
in your submissions.
