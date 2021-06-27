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

        nvars = m.getNVars()
        nconss = m.getNConss()
        nintvars = m.getNIntVars()
        nbinvars = m.getNBinVars()

        observation = (nvars, nconss, nintvars, nbinvars)

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
        nvars, nconss, nintvars, nbinvars = observation

        scip_params = {}  # default SCIP parameters

        action = scip_params

        return action
