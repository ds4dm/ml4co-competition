## Machine Learning for Combinatorial Optimization - NeurIPS 2021 Competition

(please cite the following paper)

**[The Machine Learning for Combinatorial Optimization Competition (ML4CO): Results and Insights](https://proceedings.mlr.press/v176/gasse22a.html)** Maxime Gasse, Simon Bowly, Quentin Cappart, Jonas Charfreitag, Laurent Charlin, Didier Chételat, Antonia Chmiela, Justin Dumouchelle, Ambros Gleixner, Aleksandr M. Kazachkov, Elias Khalil, Pawel Lichocki, Andrea Lodi, Miles Lubin, Chris J. Maddison, Morris Christopher, Dimitri J. Papageorgiou, Augustin Parjadis, Sebastian Pokutta, Antoine Prouvost, Lara Scavuzzo, Giulia Zarpellon, Linxin Yang, Sha Lai, Akang Wang, Xiaodong Luo, Xiang Zhou, Haohan Huang, Shengcheng Shao, Yuanming Zhu, Dong Zhang, Tao Quan, Zixuan Cao, Yang Xu, Zhewei Huang, Shuchang Zhou, Chen Binbin, He Minggui, Hao Hao, Zhang Zhiyu, An Zhiwu, Mao Kun, Proceedings of the NeurIPS 2021 Competitions and Demonstrations Track, PMLR 176:220-231, 2022.

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

### Final evaluation

During the competition we will evaluate submissions
on a weekly basis, and update our online leaderboards
while the competition is running. Participants do not have
to send a submission every week, but are encouraged to submit
regularly to make the competition a live event. In order to
prevent test overfitting, those intermediate evaluations will
be performed on a fixed subset of the test set (20%),
while only the final evaluation, which will tell the winners, will be
performed on the entire test set.

### Additional remarks

We provide an official support to participants via the
[Github discussions](https://github.com/ds4dm/ml4co-competition/discussions)
feature. Please direct any technical or general question
regarding the competition there, and feel free to answer
the questions of other participants as well. We will not provide a
privileged support to any of the participants, except in situations where
it concerns information about their submission which they do not want to share.
To contact us directly, use the competition's email adress: [ml4co.competition@gmail.com
](mailto:ml4co.competition@gmail.com).

We will not run the training of your ML models. Please send us
only your final, pre-trained model, ready for evaluation.

We only offer support for Linux and MacOS, the Windows platform being
currently unsupported by Ecole.

### Sponsors

We thank [Compute-Canada](https://www.computecanada.ca/), [Calcul Québec](https://www.calculquebec.ca/en/) and
[Westgrid](https://www.westgrid.ca/) for providing the infrastructure and compute ressources that allow us to
run the competition.
