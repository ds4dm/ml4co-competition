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


class TimeLimitDualIntegral(ecole.reward.DualIntegral):

    def __init__(self, offset=None, initial_dual_bound=None):
        self._offset = offset
        self._initial_dual_bound = initial_dual_bound

        super().__init__(wall=True, bound_function=lambda model: (self.offset, self.initial_dual_bound))

    def before_reset(self, model):
        # trick to allow the dual bound initial value and offset to be set dynamically
        self.offset = self._offset() if callable(self._offset) else self._offset
        self.initial_dual_bound = self._initial_dual_bound() if callable(self._initial_dual_bound) else self._initial_dual_bound

        # default values if none was provided
        if self.offset is None:
            self.offset = 0.0

        if self.initial_dual_bound is None:
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
                dual_bound = -m.infinity() if m.getObjectiveSense() == "minimize" else m.infinity()
            else:
                dual_bound = m.getDualbound()

            # account for maximization problems
            if m.getObjectiveSense() == "minimize":
                reward += -(max(dual_bound, self.initial_dual_bound) - self.offset) * time_left
            else:
                reward += (min(dual_bound, self.initial_dual_bound) - self.offset) * time_left

        return reward


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


class Branching(ecole.environment.Environment):
    __Dynamics__ = BranchingDynamics
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
