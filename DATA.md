## Data description

### Training datasets

The public instances can been downloaded 
[here](https://drive.google.com/file/d/1MytdY3IwX_aFRWdoc0mMfDN9Xg1EKUuq/view?usp=sharing),
and are to placed inside the `instances` folder. The folder structure after the datasets are
set up looks as follows
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

**Important note**: for each benchmark we propose a pre-defined split of the public
instances into a training set (`train`) and a validation set (`valid`). Participants
do not have to respect this arbitrary choice, and
are free to use all the provided instances in whichever way they like without any restriction.
All the instances included in `train` and `valid` can be considered training instances.

### Test datasets

The test instances used to evaluate the submissions
will be kept hidden until the end of the competition.
```
instances/
  1_item_placement/
    test/           -> 100 instances
  2_load_balancing/
    test/           -> 100 instances
  3_anonymous/
    test/           -> 20 instances
```

### File formats

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

### Problem benchmarks

Here we give a short description of each problem benchmark. In particular,
we describe how each problem instance is modeled as a Mixed-Integer Linear
Program (MILP).

#### 1: Balanced Item Placement

There are ![formula](https://render.githubusercontent.com/render/math?math=I) items,
![formula](https://render.githubusercontent.com/render/math?math=B) bins, and
![formula](https://render.githubusercontent.com/render/math?math=R) resource types.
Each item ![formula](https://render.githubusercontent.com/render/math?math=i)
has a fixed resource requirement, for each resource type
![formula](https://render.githubusercontent.com/render/math?math=r). Each bin
![formula](https://render.githubusercontent.com/render/math?math=b) has a fixed capacity, 
for each resource type ![formula](https://render.githubusercontent.com/render/math?math=r). The goal is
to place all items in bins, while minimizing the imbalance of the resources used across all bins.

##### Constants

`Capacity_$b_$r`: the amount of resource `r` available in bin `b`

![formula](https://render.githubusercontent.com/render/math?math=\forall_{b,r},\quad%20\textit{Capacity}_{b,r}%20\in%20\mathbb{R}_{\geq%200})

`Size_$i_$r`: the amount of resource `r` required by item `i`

![formula](https://render.githubusercontent.com/render/math?math=\forall_{i,r},\quad%20\textit{Size}_{i,r}%20\in%20\mathbb{R}_{\geq%200})

##### Decision variables

`place_$i_$b`: a binary variable for indicating whether to place item `i` in bin `b`.

![formula](https://render.githubusercontent.com/render/math?math=\forall_{i,b},\quad%20\textit{place}_{i,b}%20\in%20\\{0,1\\})

##### Implicit decision variables

`deficit_$b_$r`: a continuous variable between 0 and 1 for tracking the normalized imbalance of resource `r` in bin `b`.

![formula](https://render.githubusercontent.com/render/math?math=\forall_{b,r},\quad%20\textit{deficit}_{b,r}%20\in%20[0,1])

`max_deficit_$r`: a continuous variable between 0 and 1 for tracking the max normalized imbalance of resource `r` across all bins.

![formula](https://render.githubusercontent.com/render/math?math=\forall_{r},\quad%20\textit{max\\_deficit}_{r}%20\in%20[0,1])

##### Constraints

`copies_ct_$i`: all items must be placed once.

![formula](https://render.githubusercontent.com/render/math?math=\forall_i,\quad%20\sum_b%20\textit{place}_{i,b}%20=%201)

`supply_ct_$b_$r`: bin capacities must be respected.

![formula](https://render.githubusercontent.com/render/math?math=\forall_{b,r},\quad%20\sum_i%20\textit{Size}_{i,r}%20\times%20\textit{place}_{i,b}%20\leq%20\textit{Capacity}_{b,r})

`deficit_ct_$b_$r`: normalized imbalance of resources is tracked for each bin and resource.

![formula](https://render.githubusercontent.com/render/math?math=\forall_{b,r},\quad%201%20-%20\frac{B}{\sum_i%20\textit{Size}_{i,r}}\sum_i%20\textit{Size}_{i,r}%20\times%20\textit{place}_{i,b}%20=%20\textit{deficit}_{b,r})

`max_deficit_ct_$r`: max normalized imbalance of resources across all bins is tracked for each resource.

![formula](https://render.githubusercontent.com/render/math?math=\forall_{b,r},\quad%20\textit{deficit}_{b,r}%20\leq%20\textit{max\\_deficit}_{r})

##### Objective

Minimize the imbalance of resources used across all bins.

![formula](https://render.githubusercontent.com/render/math?math=\text{minimize}\quad%2010\times%20B\times%20R%20\sum_r%20\textit{max\\_deficit}_{r}%2B\sum_{b,r}\textit{deficit}_{b,r})


#### 2: Workload Apportionment



#### 3: Anonymous Problem

The third problem benchmark is anonymous, and thus we do not provide a description of the problem instances.
