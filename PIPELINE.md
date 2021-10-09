### Evaluation pipeline

Team submissions will be evaluated within an Ubuntu-based singularity container,
in order to isolate any side-effect from the code's execution on the host. As such, we provide
the exact singularity image recipe and scripts that we use for evaluation, and we encourage
participants to test their code (installation + evaluation) within the same pipeline
before they make a submission.

#### Singularity image set-up (only once)

Make sure `singularity` is installed and available. Then, build the Singularity image.
```bash
sh singularity/01_singularity_build.sh
```

This script is configured to do the build remotely (`--remote`), which requires to create
a [Sylab account](https://cloud.sylabs.io/home) and register a token. Alternatively, you can also remove the
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
bash singularity/03_participant_run.sh YOUR_TEAM_NAME primal item_placement
bash singularity/03_participant_run.sh YOUR_TEAM_NAME primal load_balancing
bash singularity/03_participant_run.sh YOUR_TEAM_NAME primal anonymous

# Dual task
bash singularity/03_participant_run.sh YOUR_TEAM_NAME dual item_placement
bash singularity/03_participant_run.sh YOUR_TEAM_NAME dual load_balancing
bash singularity/03_participant_run.sh YOUR_TEAM_NAME dual anonymous

# Config task
bash singularity/03_participant_run.sh YOUR_TEAM_NAME config item_placement
bash singularity/03_participant_run.sh YOUR_TEAM_NAME config load_balancing
bash singularity/03_participant_run.sh YOUR_TEAM_NAME config anonymous
```

**Note**: additional argument such as `--timelimit T` or `--debug` can also be provided here,
and will be passed to the Python evaluation script.
