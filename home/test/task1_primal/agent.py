import ecole as ec
import numpy as np


class ObservationFunction():

    def __init__(self, problem):
        self.problem = problem  # to devise problem-specific observations

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        if done:
            return None

        m = model.as_pyscipopt()
        variables = m.getVars()
        inf = m.infinity()

        lbs = np.asarray([v.getLbLocal() for v in variables])
        ubs = np.asarray([v.getUbLocal() for v in variables])
        has_lb = np.asarray([lb > -inf for lb in lbs])
        has_ub = np.asarray([ub < inf for ub in ubs])

        observation = (has_lb, has_ub, lbs, ubs)

        return observation


class Policy():

    def __init__(self, problem):
        self.rng = np.random.RandomState()
        self.problem = problem  # to devise problem-specific policies

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        # called before an episode starts
        pass

    def __call__(self, action_set, observation):
        has_lb, has_ub, lbs, ubs = observation 

        # safety check: make sure all variables in the action set have lower and upper bounds
        assert all(has_lb[action_set]) and all(has_ub[action_set])

        # decide on (partial) variable assignments
        var_ids = action_set
        var_vals = self.rng.randint(lbs[var_ids], ubs[var_ids]+1)  # random sampling

        action = (var_ids, var_vals)

        return action
