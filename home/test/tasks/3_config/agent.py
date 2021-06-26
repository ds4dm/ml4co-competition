import ecole as ec
import numpy as np


class ObservationFunction():

    def __init__(self, problem):
        self.problem = problem  # to devise problem-specific observations
        self.milp_bipartite = ec.observation.MilpBipartite()

    def before_reset(self, model):
        self.milp_bipartite.before_reset(model)

    def extract(self, model, done):
        if done:
            return None

        observation = self.milp_bipartite.extract(model, done)

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
        milp_bipartite = observation 

        scip_params = {}  # default SCIP parameters

        action = scip_params

        return action
