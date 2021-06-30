# Getting started

Clone this repository
```bash
git clone git@github.com:ds4dm/ml4co-competition.git
cd ml4co-competition
```

Download the training instances [here](https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view?usp=sharing),
and extract the archive at the root of this repo
```bash
tar -xzf instances.tar.gz
```

Decide on a team name, and copy-paste the example contribution
```bash
cp -r submissions/example submissions/YOUR_TEAM_NAME
cd submissions/YOUR_TEAM_NAME
```

Set-up your Python dependencies by editing the following files
```
conda.yaml
init.sh
```

Install the `ml4co` conda environment by running
```bash
source init.sh
```

## Implement your agent

To compete in each of our three tasks, you just have to edit
one separate file
```
agents/
  primal.py    -> your team's agent for solving the primal task
  dual.py      -> your team's agent for solving the dual task
  config.py    -> your team's agent for solving the config task
```

If you plan to compete in a single task only, say the dual task, you'll
only have to edit the `agents/dual.py` file.

You can also include any supplementary file which is required for your
code to operate, such as additional python files, or a file containing
your machine learning model's parameters for example.

**Note**: when we will evaluate your agents, the `submissions/YOUR_TEAM_NAME`
folder will be the working directory, so that a file `submissions/YOUR_TEAM_NAME/xxx`
can be directly accessed via
```Python
with open("xxx") as f:
  do_something
```

Each of the `agents/primal.py`, `agents/dual.py` and `agents/config.py` files work
in the same way. Inside you will find two things:
 - an [`ObservationFunction`](https://doc.ecole.ai/py/en/stable/reference/observations.html) that
defines which features you want to extract from the SCIP
solver at each step of the decision process. See the
[Ecole documentation](https://doc.ecole.ai/py/en/stable/howto/create-functions.html)
to learn how to create a new observation function from scratch using
[PySCIPOpt](https://github.com/scipopt/PySCIPOpt),
or to build one by reusing existing observation functions already built in Ecole.
 - a Policy responsible for taking decisions given past observations.

**Note**: both the observation function and the policy have a `seed(seed)` method,
which is to be used to make your code deterministic. Also, each observation function
or policy receives a `problem` string which indicates the problem benchmark currently
being process. This string can be used to devise problem-specific agents.

## Evaluate your agent

You can evaluate the performance of your agent locally on the
validation instances (`valid` folder) of each problem benchmark by running

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

The results will be saved to the following csv files:
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

## Submit your agent

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

Please send us such folder compressed into a single ZIP file when you make
a submission, and also please tell us whether your code requires a GPU or not
to run.

## Additional remarks

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

To make sure that your submission will install and run on our side,
please try to evaluate it on your side within the [singularity](https://sylabs.io/docs/)
image we provide.
