import ecole as ec
import numpy as np
import glob
import argparse

from environment import RootPrimalSearch
from agent import MyObservationFunction, MyPolicy


class Info():

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        m = model.as_pyscipopt()
        return {'primal_bound': m.getPrimalbound(),
                'dual_bound': m.getDualbound(),
                'nlpiters': m.getNLPIterations(),
                'nnodes': m.getNNodes(),
                'solvingtime': m.getSolvingTime()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='Problem benchmark to process.',
        choices=['item_placement', 'load_balancing', 'anonymous'],
    )
    args = parser.parse_args()

    if args.problem == 'item_placement':
        instances = glob.glob('instances/1_item_placement/test/*.mps.gz')
    elif args.problem == 'load_balancing':
        instances = glob.glob('instances/2_load_balancing/test/*.mps.gz')
    elif args.problem == 'anonymous':
        instances = glob.glob('instances/3_anonymous/test/*.mps.gz')

    time_limit = 10  # 5*60

    # environment
    env = RootPrimalSearch(
        time_limit=time_limit,
        observation_function=MyObservationFunction(problem=args.problem),
        reward_function=-ec.reward.PrimalIntegral(wall=True),  # minimization <=> negative reward
        information_function=Info()  # TODO: remove (debugging only)
    )

    # agent
    policy = MyPolicy(problem=args.problem)

    # evaluation loop
    for seed, instance in enumerate(instances):
        policy.seed(seed)
        env.seed(seed)

        print(f"instance={instance} seed={seed}")

        policy.reset()
        observation, action_set, reward, done, info = env.reset(instance)
        print(info)

        cum_reward = 0  # discard initial reward
        while not done:
            action = policy(action_set, observation)
            observation, action_set, reward, done, info = env.step(action)
            print(info)

            cum_reward += reward
        print(f"Cumulated reward: {cum_reward}")

