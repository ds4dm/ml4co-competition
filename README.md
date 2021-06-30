### Machine Learning for Combinatorial Optimization - NeurIPS 2021 Competition

This repository contains the base code that supports the competition, as well as
some code examples and baseline implementations for each of the three decision tasks
that participants can compete in (`primal`, `dual`, `config`).

**[Official website](https://www.ecole.ai/2021/ml4co-competition/)**: competition guidelines, team registration, rules, and leaderboard.

**[Getting Started](START.md)**: get the data, implement and evaluate your agent, make a submission.

**More information**:

 - **[Data description](DATA.md)**: the three datasets (`item_placement`, `load_balancing`, `anonymous`) and the data files.

 - **[Tasks description](TASKS.md)**: implementation details about the three environments (`primal`, `dual`, `config`) and reward functions.

 - **[Evaluation pipeline](SINGULARITY.md)**: before you submit, make sure your code installs and runs within our singularity container.

 - **[Evaluation platform](EVALUATION.md)**: the hardware, software, and OS specifications of the platform your code will be evaluated on.

 - **Documentation**: **[Ecole](https://doc.ecole.ai/)** - **[SCIP](https://scipopt.org/doc/html/)** - **[PySCIPOpt](https://scipopt.github.io/PySCIPOpt/docs/html/)**

We thank [Compute-Canada](https://www.computecanada.ca/), [Calcul Québec](https://www.calculquebec.ca/en/) and
[Westgrid](https://www.westgrid.ca/) for providing the infrastructure and compute ressources that allow us to
run the competition.

## 2. File structure

Files in this repo are organized as follows
```
instances/    -> the datasets
common/       -> the common code base of the competition, i.e., the environment, reward, and evaluation scripts
submissions/  -> the team submissions
  example/    -> an example submission
singularity/  -> the singularity image and scripts of our evaluation pipeline
```

The competition's evaluation pipeline is based on Singularity 3.7, Python 3.7, and `conda+pip`
to install the software dependencies.

### 2.1. Datasets

After the trainng instances have been downloaded 
[here](https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view?usp=sharing),
and placed inside the `instances` folder, the dataset structure should look like
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

### 2.2. Common code base

The Python scripts which are used to evaluate a submission are common to every
participant, and can be found in the `common` folder
```
common/
  environments.py -> definition of the POMDP environments for each task
  rewards.py      -> definition of the reward functions for each task
  evaluate.py     -> evaluation script
```

### 2.3. Submissions

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
See [gettng started](START.md) for help on how to make a submission.

### 2.4. Singularity scripts

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

**Note**: additional argument such as `--timelimit T` or `--debug` can also be provided here,
and will be passed to the Python evaluation script.


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
