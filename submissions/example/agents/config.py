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

        # extract the number of variables, constraints, integer variables, and binary variables of the instance
        nvars = m.getNVars()
        nconss = m.getNConss()
        nintvars = m.getNIntVars()
        nbinvars = m.getNBinVars()

        observation = (nvars, nconss, nintvars, nbinvars)

        return observation


class Policy():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific policies

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def __call__(self, action_set, observation):
        nvars, nconss, nintvars, nbinvars = observation

        if self.problem == "item_placement":
            scip_params = {
                    "presolving/maxrounds": 0,
            }  # deactivate presolving

        elif self.problem == "load_balancing":
            scip_params = {
                    "separating/maxrounds": 0,
                    "separating/maxroundsroot": 0,
            }  # deactivate cuts

        elif self.problem == "anonymous":
            scip_params = {}  # default SCIP parameters

        else:
            raise ValueError(f"Problem {self.problem} unknown.")

        action = scip_params

        return action
