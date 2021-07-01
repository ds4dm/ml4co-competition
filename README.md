## Machine Learning for Combinatorial Optimization - NeurIPS 2021 Competition

**[Official website](https://www.ecole.ai/2021/ml4co-competition/)**: competition guidelines, team registration, rules, and leaderboard.


This repository contains the base code that supports the competition, as well as
some code examples and baseline implementations for each of the three decision tasks
that participants can compete in (`primal`, `dual`, `config`). It is organized as follows
```
instances/    -> the datasets
common/       -> the common code base, i.e., environments, rewards and evaluation scripts
submissions/  -> the team submissions
  example/    -> an example submission
singularity/  -> the singularity image and scripts of our evaluation pipeline
```

### Documentation

 - **[Getting Started](START.md)**: get the data, implement and evaluate your agent, make a submission.

 - **[Data description](DATA.md)**: the three datasets (`item_placement`, `load_balancing`, `anonymous`) and the data files.

 - **[Tasks description](TASKS.md)**: how we implemented each task (`primal`, `dual`, `config`).

 - **[Evaluation pipeline](PIPELINE.md)**: make sure your code installs and runs within our pipeline before you submit.

 - **[Evaluation platform](PLATFORM.md)**: the hardware and software specifications of the platform your code will be evaluated on.

 - **APIs**: **[Ecole](https://doc.ecole.ai/)** - **[SCIP](https://scipopt.org/doc/html/)** - **[PySCIPOpt](https://scipopt.github.io/PySCIPOpt/docs/html/)**

### Additional remarks

We will not run the training of your ML models. Please send us
only your final, pre-trained model, ready for evaluation.

We provide an official support to participants via the
[Github discussions](https://github.com/ds4dm/ml4co-competition/discussions)
feature. Please direct any technical or general question
regarding the competition there, and feel free to answer
the questions of other participants as well. We will not provide a
privileged support to any of the participants, except in situations where
it concerns a detail about their submission which they do not want to share.
To contact us directly, use the competition's email adress: [ml4co.competition@gmail.com
](mailto:ml4co.competition@gmail.com
).

We only offer support for Linux and MacOS, the Windows platform being
currently unsupported by Ecole.

### Sponsors

We thank [Compute-Canada](https://www.computecanada.ca/), [Calcul Québec](https://www.calculquebec.ca/en/) and
[Westgrid](https://www.westgrid.ca/) for providing the infrastructure and compute ressources that allow us to
run the competition.
