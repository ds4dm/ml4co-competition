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


class DefaultInformationFunction():

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        m = model.as_pyscipopt()
        return {'primal_bound': m.getPrimalbound(),
                'dual_bound': m.getDualbound(),
                'nlpiters': m.getNLPIterations(),
                'nnodes': m.getNNodes(),
                'solvingtime': m.getSolvingTime()}


class RootPrimalSearch(ecole.environment.Environment):
    __Dynamics__ = RootPrimalSearchDynamics
    __DefaultInformationFunction__ = DefaultInformationFunction

    def reset(self, instance, objective_limit=None, *dynamics_args, **dynamics_kwargs):
        """We add one optional parameter not supported by Ecole yet: the instance's objective limit."""
        self.can_transition = True
        try:
            if isinstance(instance, ecole.core.scip.Model):
                self.model = instance.copy_orig()
            else:
                self.model = ecole.core.scip.Model.from_file(instance)
            self.model.set_params(self.scip_params)

            # >>> changes specific to this environment
            self.model.set_params({
                'estimation/restarts/restartpolicy': 'n',  # disable restarts
            })

            if objective_limit is not None:
                self.model.as_pyscipopt().setObjlimit(objective_limit)
            # <<<

            self.dynamics.set_dynamics_random_state(self.model, self.random_engine)

            self.observation_function.before_reset(self.model)
            self.reward_function.before_reset(self.model)
            self.information_function.before_reset(self.model)
            done, action_set = self.dynamics.reset_dynamics(
                self.model, *dynamics_args, **dynamics_kwargs
            )

            observation = self.observation_function.extract(self.model, done)
            reward_offset = self.reward_function.extract(self.model, done)
            information = self.information_function.extract(self.model, done)
            return observation, action_set, reward_offset, done, information
        except Exception as e:
            self.can_transition = False
            raise e
