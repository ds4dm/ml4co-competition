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


class RootPrimalSearchDynamics(ecole.dynamics.PrimalSearchDynamics):

    def __init__(self, time_limit, n_trials=-1):
        super().__init__(trials_per_node=n_trials, depth_freq=1, depth_start=0, depth_stop=0)  # only at the root node
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


class BranchingDynamics(ecole.dynamics.BranchingDynamics):

    def __init__(self, time_limit):
        super().__init__(pseudo_candidates=True)
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


class ConfiguringDynamics(ecole.dynamics.ConfiguringDynamics):

    def __init__(self, time_limit):
        super().__init__()
        self.time_limit = time_limit

    def reset_dynamics(self, model):
        pyscipopt_model = model.as_pyscipopt()

        # process the root node
        done, action_set = super().reset_dynamics(model)

        # set time limit after reset
        reset_time = pyscipopt_model.getSolvingTime()
        pyscipopt_model.setParam("limits/time", self.time_limit + reset_time)

        return done, action_set

    def step_dynamics(self, model, action):
        forbidden_params = [
            "limits/time",
            "timing/clocktype",
            "timing/enabled",
            "timing/reading",
            "timing/rareclockcheck",
            "timing/statistictiming",
            "limits/memory"]

        for param in forbidden_params:
            if param in action:
                raise ValueError(f"Setting the SCIP parameter '{param}' is forbidden.")

        done, action_set = super().step_dynamics(model, action)

        return done, action_set


class ObjectiveLimitEnvironment(ecole.environment.Environment):

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

            # Reset data extraction functions
            self.reward_function.before_reset(self.model)
            self.observation_function.before_reset(self.model)
            self.information_function.before_reset(self.model)

            # Place the environment in its initial state
            done, action_set = self.dynamics.reset_dynamics(
                self.model, *dynamics_args, **dynamics_kwargs
            )
            self.can_transition = not done

            # Extract additional data to be returned by reset
            reward_offset = self.reward_function.extract(self.model, done)
            if not done:
                observation = self.observation_function.extract(self.model, done)
            else:
                observation = None
            information = self.information_function.extract(self.model, done)

            return observation, action_set, reward_offset, done, information
        except Exception as e:
            self.can_transition = False
            raise e

class RootPrimalSearch(ObjectiveLimitEnvironment):
    __Dynamics__ = RootPrimalSearchDynamics
    __DefaultInformationFunction__ = DefaultInformationFunction

class Branching(ObjectiveLimitEnvironment):
    __Dynamics__ = BranchingDynamics
    __DefaultInformationFunction__ = DefaultInformationFunction

class Configuring(ObjectiveLimitEnvironment):
    __Dynamics__ = ConfiguringDynamics
    __DefaultInformationFunction__ = DefaultInformationFunction

