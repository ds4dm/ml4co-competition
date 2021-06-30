### Tasks description

Here we provide implementation details about the three environments (`primal`, `dual`, `config`) and reward functions.

#### Common code base

The Python scripts which define the environments, rewards, and evaluation loop are common to every
participant, and can be found in the `common` folder
```
common/
  environments.py -> definition of the POMDP environments for each task
  rewards.py      -> definition of the reward functions for each task
  evaluate.py     -> evaluation script
```

For all three `primal`, `dual` and `config` tasks we overload Ecole's environment class to add an optional
`objective_limit` argument in `reset(instance, objective_limit)`, so that the initial primal bound of each
instance is always used as an objective limit by SCIP (no feasible solution worst that this value will be accepted by SCIP).
```python
class ObjectiveLimitEnvironment(ecole.environment.Environment):

    def reset(self, instance, objective_limit=None, *dynamics_args, **dynamics_kwargs):
...
```

#### Primal task

The details of our primal task implementation can be found in the `RootPrimalSearchDynamics` class, which extends
Ecole's [`PrimalSearchDynamics`](https://doc.ecole.ai/py/en/stable/reference/environments.html#ecole.dynamics.PrimalSearchDynamics).
In a nutshell, this class defines the following environment dynamics:
 - SCIP restarts are deactivated
 - SCIP's default primal heuristics are deactivated
 - SCIP's time limit is set after the root node has been processed
 - initial state: SCIP processes the instance until the root node LP is solved
(includes preprocessing, cuts etc.), then applies a time limit relative to this time,
and enters an infinite loop.
 - steps: the user provides a partial solution in the form of two
lists: `(variable_ids, variable_values)`. SCIP solves the LP relaxation
with those variables fixed, and then tries to add the resulting LP solution as
a primal MILP solution.
 - terminal state: the time limit has been reached, or SCIP managed to prove optimality (which should not happen).

```python
class RootPrimalSearchDynamics(ecole.dynamics.PrimalSearchDynamics):

    def __init__(self, time_limit, n_trials=-1):
        super().__init__(trials_per_node=n_trials, depth_freq=1, depth_start=0, depth_stop=0)  # only at the root node
        self.time_limit = time_limit

    def reset_dynamics(self, model):
        pyscipopt_model = model.as_pyscipopt()

        # disable SCIP heuristics
        pyscipopt_model.setHeuristics(pyscipopt.scip.PY_SCIP_PARAMSETTING.OFF)

        # disable restarts
        model.set_params({
            'estimation/restarts/restartpolicy': 'n',
        })

        # process the root node
        done, action_set = super().reset_dynamics(model)

        # set time limit after reset
        reset_time = pyscipopt_model.getSolvingTime()
        pyscipopt_model.setParam("limits/time", self.time_limit + reset_time)

        return done, action_set
```
