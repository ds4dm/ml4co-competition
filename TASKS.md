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

For all three tasks we overload Ecole's environment class to add an optional
`objective_limit` argument in `reset(instance, objective_limit)`, so that the initial primal bound of each
instance is always used as an objective limit by SCIP (no feasible solution worst that this value will be accepted by SCIP).
```python
class ObjectiveLimitEnvironment(ecole.environment.Environment):

    def reset(self, instance, objective_limit=None, *dynamics_args, **dynamics_kwargs):
...
```

For all three tasks, the reward function share the same API and the same global behaviour. If the solver stops
before the time limit (the `limits/time` SCIP parameter), then the integration continues over the remaining time.
```python
time_left = max(m.getParam("limits/time") - m.getSolvingTime(), 0)
```
Each reward function is equipped with a `set_parameters()` method, so that we can provide the initial primal
and dual bounds of each instance in order to compute the integrals unambiguously.
```python
  def set_parameters(self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None):
      ...
```
Note that an optional `objective_offset` can be provided as well to cimpute the integrals, although this feature is not
used in the evaluation computations (`objective_offset=0`).

#### Primal task's environment

The details of our primal task implementation can be found in the `RootPrimalSearchDynamics` class, which extends
Ecole's [`PrimalSearchDynamics`](https://doc.ecole.ai/py/en/stable/reference/environments.html#ecole.dynamics.PrimalSearchDynamics).
In a nutshell, this class defines the following environment dynamics:
 - SCIP restarts are deactivated
 - SCIP's default primal heuristics are deactivated
 - SCIP's time limit is set after the root node has been processed
 - initial state: SCIP processes the instance until the root node LP is solved
(includes preprocessing, cuts etc.), then applies a time limit relative to this time,
and enters an infinite loop.
 - steps: the environment receives from the agent a partial solution in the form of two
lists, `(variable_ids, variable_values)`. SCIP solves the LP relaxation
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

#### Dual task's environment

The details of our dual task implementation can be found in the `BranchingDynamics` class, which extends
Ecole's own [`BranchingDynamics`](https://doc.ecole.ai/py/en/stable/reference/environments.html#ecole.dynamics.BranchingDynamics).
In a nutshell, this class defines the following environment dynamics:
 - SCIP restarts are deactivated
 - SCIP's primal heuristics are deactivated
 - SCIP's time limit is set after the root node has been processed
 - initial state: SCIP processes the instance until the root node LP is solved
(includes preprocessing, cuts etc.), then applies a time limit relative to this time,
and resumes solving until a branching decision has to be made.
 - steps: the environment receives from the agent a branching candidate in the form of a
non-fixed integer variable's LP column position, `var_lppos`. SCIP branches on that
variable, and then continues solving until the next branching decision has to be made.
 - terminal state: the time limit has been reached, or SCIP managed to prove optimality.

```python
class BranchingDynamics(ecole.dynamics.BranchingDynamics):

    def __init__(self, time_limit):
        super().__init__()
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

#### Config task's environment

The details of our config task implementation can be found in the `ConfiguringDynamics` class, which extends
Ecole's own [`ConfiguringDynamics`](https://doc.ecole.ai/py/en/stable/reference/environments.html#ecole.dynamics.ConfiguringDynamics).
In a nutshell, this class defines the following environment dynamics:
 - SCIP's time limit is set after the instance has been loaded
 - initial state: SCIP loads the instance.
 - steps: the environment receives from the agent a set of SCIP parameters in the form of a
dictionnary, `scip_params`. SCIP sets those parameters, and then continues solving.
 - terminal state: the time limit has been reached, or SCIP managed to prove optimality.

**Note**: it is forbidden for agents to set parameters that affect the time limit in any way.

```python
class ConfiguringDynamics(ecole.dynamics.ConfiguringDynamics):

    def __init__(self, time_limit):
        super().__init__()
        self.time_limit = time_limit

    def reset_dynamics(self, model):
        pyscipopt_model = model.as_pyscipopt()

        # process the root node
        done, action_set = super().reset_dynamics(model)

        # set time limit after reset
        reset_time = pyscipopt_model.getSolvingTime()
        pyscipopt_model.setParam("limits/time", self.time_limit + reset_time)

        return done, action_set

    def step_dynamics(self, model, action):
        forbidden_params = [
            "limits/time",
            "timing/clocktype",
            "timing/enabled",
            "timing/reading",
            "timing/rareclockcheck",
            "timing/statistictiming"]

        for param in forbidden_params:
            if param in action:
                raise ValueError(f"Setting the SCIP parameter '{param}' is forbidden.")

        done, action_set = super().step_dynamics(model, action)

        return done, action_set
```

#### Primal task's reward

The reward for the primal task is the primal bound integral (see
description [here](https://www.ecole.ai/2021/ml4co-competition/#metrics)).
Implementation details can be found in the `TimeLimitPrimalIntegral` class,
which extends Ecole's [`PrimalIntegral`](https://doc.ecole.ai/py/en/stable/reference/rewards.html#ecole.reward.PrimalIntegral).
In a nutshell, it works as follows:
 - the `set_parameters()` method is called before each instance is
processed, and sets the instance's initial primal bound.
 - the primal bound used to perform the mathematical integration is
the initial primal bound, or the primal bound found by the SCIP solver
if it improves upon the initial primal bound.
 - if the episode stops before the time limit is reached, the integration
continues over the remaining time window (`max(m.getParam("limits/time") - m.getSolvingTime(), 0)`)
and is incorporated into the final reward.

```python
class TimeLimitPrimalIntegral(ecole.reward.PrimalIntegral):
...
```

#### Dual task's reward

The reward for the dual task is the dual bound integral (see
description [here](https://www.ecole.ai/2021/ml4co-competition/#metrics)).
Implementation details can be found in the `TimeLimitDualIntegral` class,
which extends Ecole's [`DualIntegral`](https://doc.ecole.ai/py/en/stable/reference/rewards.html#ecole.reward.DualIntegral).
In a nutshell, it works as follows:
 - the `set_parameters()` method is called before each instance is
processed, and sets the instance's initial dual bound.
 - the dual bound used to perform the mathematical integration is
the initial dual bound, or the dual bound found by the SCIP solver
if it improves upon the initial dual bound.
 - if the episode stops before the time limit is reached, the integration
continues over the remaining time window (`max(m.getParam("limits/time") - m.getSolvingTime(), 0)`)
and is incorporated into the final reward.

```python
class TimeLimitDualIntegral(ecole.reward.DualIntegral):
...
```

#### Config task's reward

The reward for the config task is the primal-dual bound integral (see
description [here](https://www.ecole.ai/2021/ml4co-competition/#metrics)).
Implementation details can be found in the `TimeLimitPrimalDualIntegral` class,
which extends Ecole's [`PrimalDualIntegral`](https://doc.ecole.ai/py/en/stable/reference/rewards.html#ecole.reward.PrimalDualIntegral).
In a nutshell, it works as follows:
 - the `set_parameters()` method is called before each instance is
processed, and sets the instance's initial primal and dual bounds.
 - the primal and dual bounds used to perform the mathematical integration are
the initial primal and dual bounds, or the bounds found by the SCIP solver
if they improves upon the initial ones.
 - if the episode stops before the time limit is reached, the integration
continues over the remaining time window (`max(m.getParam("limits/time") - m.getSolvingTime(), 0)`)
and is incorporated into the final reward.

```python
class TimeLimitPrimalDualIntegral(ecole.reward.PrimalDualIntegral):
```
