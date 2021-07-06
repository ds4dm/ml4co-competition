import ecole as ec
import numpy as np


class ObservationFunction():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific observations

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def before_reset(self, model):
        # called when a new episode is about to start
        pass

    def extract(self, model, done):
        if done:
            return None

        m = model.as_pyscipopt()
        variables = m.getVars(transformed=True)
        inf = m.infinity()

        # extract the upper and lower bounds of all variables
        lbs = np.asarray([v.getLbLocal() for v in variables])
        ubs = np.asarray([v.getUbLocal() for v in variables])
        has_lb = np.asarray([lb > -inf for lb in lbs])
        has_ub = np.asarray([ub < inf for ub in ubs])

        observation = (has_lb, has_ub, lbs, ubs)

        return observation


class Policy():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.rng = np.random.RandomState()
        self.problem = problem  # to devise problem-specific policies

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        self.rng = np.random.RandomState(seed)

    def __call__(self, action_set, observation):
        has_lb, has_ub, lbs, ubs = observation 

        # safety check: make sure all variables in the action set have lower and upper bounds
        assert all(has_lb[action_set]) and all(has_ub[action_set])

        # decide on (partial) variable assignments
        var_ids = action_set
        var_vals = self.rng.randint(lbs[var_ids], ubs[var_ids]+1)  # random sampling

        action = (var_ids, var_vals)

        return action
