# Evaluation pipeline for the ML4CO competition

The idea is that we have a separate `home/TEAM` folder for each team, where we put all of their files. They can edit the `conda.yaml` file to install additional conda or pip packages, and optionally the `init.sh` file if they want to install dependencies manually. A minimal example is in `home/test/`.

## Singularity set-up

Load the Singularity module (must be run every time at login, Compute-Canada only)
```bash
source 00_compute_canada.sh
```

Build the Singularity image (only once). The script is configured to do the build remotely (--remote), which requires to create a [Sylab account](https://cloud.sylabs.io/home).
```bash
sh 01_singularity_build.sh
```

Note: at the moment the `data/instances` directory is absent from the repo. On the competition's cluster it has to be linked as follows:
```bash
ln -s /project/def-sponsor00/ml4co-competition/instances data/instances
```

## Team evaluation pipeline

### Team set up (only once per team)

Decide on a team to evaluate, and place their submission in a `home/TEAM` folder.
```bash
TEAM=test
```

Set up an `ml4co` conda environment within the team's container, based on the team's `conda.yaml` and `init.sh` files. Requires internet access to download dependencies.
```bash
sh 02_participant_init.sh $TEAM
```

### Team evaluation

Decide on a task and a problem benchmark to evaluate.
```bash
TASK=primal  # primal, dual, config
PROBLEM=item_placement  # item_placement, load_balancing, anonymous
```

Run the evalution script within the team's container. No internet access.
```bash
sh 03_participant_run.sh $TEAM $TASK $PROBLEM
```
