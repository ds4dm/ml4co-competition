
The idea is that we have a separate `home/TEAM_NAME` folder for each team, where we put all of their files. They can edit the `environment.yaml` file to install additional conda packages, and optionally the `init.sh` file if they want to install non-conda dependencies (e.g., `pip install ...`). A minimal example is in `home/test/`.

```bash
source 00_compute_canada.sh
sh 01_singularity_build.sh  # build the image, only once
sh 02_participant_init.sh test  # set up the team dependencies, only once per team. Internet access.
sh 03_participant_run.sh test  # run the evalution script within the team's environment. No internet access.
```

Note: at the moment the `data/instances` directory is absent from the repo. On the competition's cluster it can be recovered using a symlink:
```bash
ln -s /project/def-sponsor00/ml4co-competition/instances data/instances
```
