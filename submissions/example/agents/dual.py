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
        lp_columns = m.getLPColsData()
        lp_variables = [col.getVar() for col in lp_columns]

        # extract the current LP solution value and the objective coefficient of each variable in the current LP
        lp_var_lpsols = np.asarray([var.getLPSol() for var in lp_variables])
        lp_var_objs = np.asarray([var.getObj() for var in lp_variables])

        observation = (lp_var_lpsols, lp_var_objs)

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
        branching_candidates = action_set
        lp_var_lpsols, lp_var_objs = observation

        # choose branching variable among the candidates
        candidate_objs = lp_var_objs[branching_candidates]
        candidate_probs = np.exp(np.abs(candidate_objs)) / np.sum(np.exp(np.abs(candidate_objs)))
        branching_var = self.rng.choice(branching_candidates, p=candidate_probs)  # weighted random sampling

        action = branching_var

        return action
