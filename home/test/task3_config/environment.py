import ecole
import pyscipopt


class DefaultInformationFunction():

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        m = model.as_pyscipopt()

        stage = m.getStage()
        sense = 1 if m.getObjectiveSense() == "minimize" else -1

        primal_bound = sense * m.infinity()
        dual_bound = sense * -m.infinity()
        nlpiters = 0
        nnodes = 0
        solvingtime = 0
        status = m.getStatus()

        if stage >= pyscipopt.scip.PY_SCIP_STAGE.PROBLEM:
            primal_bound = m.getObjlimit()
            nnodes = m.getNNodes()
            solvingtime = m.getSolvingTime()

        if stage >= pyscipopt.scip.PY_SCIP_STAGE.TRANSFORMED:
            primal_bound = m.getPrimalbound()
            dual_bound = m.getDualbound()

        if stage >= pyscipopt.scip.PY_SCIP_STAGE.PRESOLVING:
            nlpiters = m.getNLPIterations()

        return {'primal_bound': primal_bound,
                'dual_bound': dual_bound,
                'nlpiters': nlpiters,
                'nnodes': nnodes,
                'solvingtime': solvingtime,
                'status': status}


class TimeLimitPrimalDualIntegral(ecole.reward.PrimalDualIntegral):

    def __init__(self, initial_primal_bound=None, initial_dual_bound=None):
        self._initial_primal_bound = initial_primal_bound
        self._initial_dual_bound = initial_dual_bound

        super().__init__(wall=True, bound_function=lambda model: (self.initial_primal_bound, self.initial_dual_bound))

    def before_reset(self, model):
        # trick to allow the primal and dual bound initial values to be set dynamically
        self.initial_primal_bound = self._initial_primal_bound() if callable(self._initial_primal_bound) else self._initial_primal_bound
        self.initial_dual_bound = self._initial_dual_bound() if callable(self._initial_dual_bound) else self._initial_dual_bound

        # default values if none was provided
        if self.initial_primal_bound is None:
            self.initial_primal_bound = model.as_pyscipopt().getObjlimit()

        if self.initial_dual_bound is None:
            m = model.as_pyscipopt()
            self.initial_dual_bound = -m.infinity() if m.getObjectiveSense() == "minimize" else m.infinity()

        super().before_reset(model)

    def extract(self, model, done):
        reward = super().extract(model, done)

        # adjust the final reward if the time limit has not been reached
        if done:
            m = model.as_pyscipopt()
            # keep integrating over the time left
            time_left = max(m.getParam("limits/time") - m.getSolvingTime(), 0)
            if m.getStage() < pyscipopt.scip.PY_SCIP_STAGE.TRANSFORMED:
                primal_bound = m.getObjlimit()
                dual_bound = -m.infinity() if m.getObjectiveSense() == "minimize" else m.infinity()
            else:
                primal_bound = m.getPrimalbound()
                dual_bound = m.getDualbound()

            # account for maximization problems
            if m.getObjectiveSense() == "minimize":
                reward += (min(primal_bound, self.initial_primal_bound) - max(dual_bound, self.initial_dual_bound)) * time_left
            else:
                reward += -(max(primal_bound, self.initial_primal_bound) - min(dual_bound, self.initial_dual_bound)) * time_left

        return reward


class ConfiguringDynamics(ecole.dynamics.ConfiguringDynamics):

    def __init__(self, time_limit):
        super().__init__()
        self.time_limit = time_limit

    def reset_dynamics(self, model):
        m = model.as_pyscipopt()

        # set the time limit
        m.setParam("limits/time", self.time_limit)

        done, action_set = super().reset_dynamics(model)

        return done, action_set

    def step_dynamics(self, model, action):
        if "limits/time" in action:
            raise ValueError("Setting the SCIP parameter 'limits/time' is forbidden.")

        done, action_set = super().step_dynamics(model, action)

        return done, action_set


class Configuring(ecole.environment.Environment):
    __Dynamics__ = ConfiguringDynamics
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

