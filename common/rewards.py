import ecole
import pyscipopt


class IntegralParameters():

    def __init__(self, offset=None, initial_primal_bound=None, initial_dual_bound=None):
        self._offset = offset
        self._initial_primal_bound = initial_primal_bound
        self._initial_dual_bound = initial_dual_bound

    def fetch_values(self, model):
        # trick to allow the parameters to be set dynamically
        self.offset = self._offset() if callable(self._offset) else self._offset
        self.initial_primal_bound = self._initial_primal_bound() if callable(self._initial_primal_bound) else self._initial_primal_bound
        self.initial_dual_bound = self._initial_dual_bound() if callable(self._initial_dual_bound) else self._initial_dual_bound

        # fetch default values if none was provided
        if self.offset is None:
            self.offset = 0.0

        if self.initial_primal_bound is None:
            self.initial_primal_bound = model.as_pyscipopt().getObjlimit()

        if self.initial_dual_bound is None:
            self.initial_dual_bound = -m.infinity() if m.getObjectiveSense() == "minimize" else m.infinity()


class TimeLimitPrimalIntegral(ecole.reward.PrimalIntegral):

    def __init__(self):
        self.parameters = IntegralParameters()
        super().__init__(wall=True, bound_function=lambda model: (
            self.parameters.offset,
            self.parameters.initial_primal_bound))

    def set_parameters(self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None):
        self.parameters = IntegralParameters(
            offset=objective_offset,
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound)

    def before_reset(self, model):
        self.parameters.fetch_values(model)
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
            else:
                primal_bound = m.getPrimalbound()

            offset = self.parameters.offset
            initial_primal_bound = self.parameters.initial_primal_bound

            # account for the model's objective direction (maximization vs minimization)
            if m.getObjectiveSense() == "minimize":
                reward += (min(primal_bound, initial_primal_bound) - offset) * time_left
            else:
                reward += -(max(primal_bound, initial_primal_bound) - offset) * time_left

        return reward


class TimeLimitDualIntegral(ecole.reward.DualIntegral):

    def __init__(self):
        self.parameters = IntegralParameters()
        super().__init__(wall=True, bound_function=lambda model: (
            self.parameters.offset,
            self.parameters.initial_dual_bound))

    def set_parameters(self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None):
        self.parameters = IntegralParameters(
            offset=objective_offset,
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound)

    def before_reset(self, model):
        self.parameters.fetch_values(model)
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

            offset = self.parameters.offset
            initial_dual_bound = self.parameters.initial_dual_bound

            # account for the model's objective direction (maximization vs minimization)
            if m.getObjectiveSense() == "minimize":
                reward += -(max(dual_bound, initial_dual_bound) - offset) * time_left
            else:
                reward += (min(dual_bound, initial_dual_bound) - offset) * time_left

        return reward


class TimeLimitPrimalDualIntegral(ecole.reward.PrimalDualIntegral):

    def __init__(self):
        self.parameters = IntegralParameters()
        super().__init__(wall=True, bound_function=lambda model: (
            self.parameters.initial_primal_bound,
            self.parameters.initial_dual_bound))

    def set_parameters(self, objective_offset=None, initial_primal_bound=None, initial_dual_bound=None):
        self.parameters = IntegralParameters(
            offset=objective_offset,
            initial_primal_bound=initial_primal_bound,
            initial_dual_bound=initial_dual_bound)

    def before_reset(self, model):
        self.parameters.fetch_values(model)
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

            initial_primal_bound = self.parameters.initial_primal_bound
            initial_dual_bound = self.parameters.initial_dual_bound

            # account for the model's objective direction (maximization vs minimization)
            if m.getObjectiveSense() == "minimize":
                reward += (min(primal_bound, initial_primal_bound) - max(dual_bound, initial_dual_bound)) * time_left
            else:
                reward += -(max(primal_bound, initial_primal_bound) - min(dual_bound, initial_dual_bound)) * time_left

        return reward

