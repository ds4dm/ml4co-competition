### Data description

The training instances can been downloaded 
[here](https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view?usp=sharing),
and placed inside the `instances` folder
```
instances/    -> the datasets
```

For each benchmark we propose a pre-defined split of the public
instances into a training set (`train`) and a validation set (`valid`).
The folder structure after the datasets are set up looks as follows
```
instances/
  1_item_placement/
    train/           -> 9900 instances
    valid/           -> 100 instances
  2_load_balancing/
    train/           -> 9900 instances
    valid/           -> 100 instances
  3_anonymous/
    train/           -> 98 instances
    valid/           -> 20 instances
```

**Important note**: participants do not have to respect this arbitrary choice, and
are free to use all the provided instances in whichever way they like without any restriction.
All the instances included in `train` and `valid` can be considered training instances.

The test instances used to evaluate the submissions
will be kept hidden until the end of the competition.

#### Instance files

Each problem instance is composed of two files which follows the same naming pattern, for instance,
```
item_placement_147.mps.gz  -> the MILP instance file in compressed MPS format
item_placement_147.json    -> a JSON file with pre-computed information about the instance
```

In the JSON files we store a pre-computed initial primal bound and initial dual bound
for each instance, which are used in the computation of our evaluation metrics. The JSON
content look as follows:
```
{"dual_bound": 4.063450550000058, "primal_bound": 671.5409895199994}
```

Those initial bounds were obtained as follows:
 - primal bound: the value of the first feasible solution found by the SCIP solver
 - dual bound: the value of the first LP relaxation solved by the SCIP solver
