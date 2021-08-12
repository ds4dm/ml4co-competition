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
        return None

class Policy():

    def __init__(self, problem):
        # called once for each problem benchmark
        self.problem = problem  # to devise problem-specific policies

    def seed(self, seed):
        # called before each episode
        # use this seed to make your code deterministic
        pass

    def reset(self):
        # called before an episode starts
        pass

    def __call__(self, action_set, observation):

        if self.problem == 'item_placement':
            scip_params = {
                'branching/clamp': 0.4057581460701715,
                'branching/lpgainnormalize': 'l',
                'branching/midpull': 0.024561156375531357,
                'branching/midpullreldomtrig': 0.44584662726953606,
                'branching/preferbinary': True,
                'branching/scorefac': 0.440795504211412,
                'branching/scorefunc': 'q',
                'lp/colagelimit': 1734636514,
                'lp/pricing': 's',
                'lp/rowagelimit': 1131389812,
                'nodeselection/childsel': 'p',
                'separating/cutagelimit': 453697232,
                'separating/maxcuts': 264981451,
                'separating/maxcutsroot': 1670303952,
                'separating/minortho': 0.7151858341872178,
                'separating/minorthoroot': 0.8521039212771241,
                'separating/poolfreq': 54298
            }
        elif self.problem == 'load_balancing':
            scip_params = {
                'branching/clamp': 0.3811269248003202,
                'branching/lpgainnormalize': 's',
                'branching/midpull': 0.3453080682951223,
                'branching/midpullreldomtrig': 0.05970995942365931,
                'branching/preferbinary': True,
                'branching/scorefac': 0.536176849053012,
                'branching/scorefunc': 's',
                'lp/colagelimit': 888577009,
                'lp/pricing': 'd',
                'lp/rowagelimit': 1027409045,
                'nodeselection/childsel': 'l',
                'separating/cutagelimit': 16983954,
                'separating/maxcuts': 688798976,
                'separating/maxcutsroot': 394234897,
                'separating/minortho': 0.24479292773399786,
                'separating/minorthoroot': 0.5665907046327899,
                'separating/poolfreq': 8764
            }
        else:
            scip_params = {
                'branching/clamp': 0.006590788269541181,
                'branching/lpgainnormalize': 's',
                'branching/midpull': 0.4161569073081126,
                'branching/midpullreldomtrig': 0.9683848340733884,
                'branching/preferbinary': False,
                'branching/scorefac': 0.9953185866847221,
                'branching/scorefunc': 's',
                'lp/colagelimit': 656974625,
                'lp/pricing': 'd',
                'lp/rowagelimit': 1327520915,
                'nodeselection/childsel': 'i',
                'separating/cutagelimit': 768058951,
                'separating/maxcuts': 863446296,
                'separating/maxcutsroot': 1930125520,
                'separating/minortho': 0.5471674120926198,
                'separating/minorthoroot': 0.5873810294839703,
                'separating/poolfreq': 8888
            }

        action = scip_params

        return action
