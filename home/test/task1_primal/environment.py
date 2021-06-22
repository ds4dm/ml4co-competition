import ecole
from pyscipopt.scip import PY_SCIP_PARAMSETTING


class RootPrimalSearchDynamics(ecole.dynamics.PrimalSearchDynamics):

    def __init__(self, time_limit):
        super().__init__(trials_per_node=-1, depth_freq=1, depth_start=0, depth_stop=-1)
        self.time_limit = time_limit

    def reset_dynamics(self, model):
        pyscipopt_model = model.as_pyscipopt()

        # disable SCIP default heuristics
        pyscipopt_model.setHeuristics(PY_SCIP_PARAMSETTING.OFF)

        # process the root node
        done, action_set = super().reset_dynamics(model)

        # set time limit after reset
        reset_time = pyscipopt_model.getSolvingTime()
        pyscipopt_model.setParam("limits/time", self.time_limit + reset_time)

        return done, action_set


class RootPrimalSearch(ecole.environment.Environment):
    __Dynamics__ = RootPrimalSearchDynamics
