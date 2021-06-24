import argparse
import csv
import json
import pathlib

import ecole as ec
import numpy as np

import environment
import agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='Problem benchmark to process.',
        choices=['item_placement', 'load_balancing', 'anonymous'],
    )
    args = parser.parse_args()

    if args.problem == 'item_placement':
        instance_files = pathlib.Path.cwd().glob('instances/1_item_placement/test/*.mps.gz')
        results_file = pathlib.Path(f"task1_primal/results/1_item_placement.csv")
    elif args.problem == 'load_balancing':
        instance_files = pathlib.Path.cwd().glob('instances/2_load_balancing/test/*.mps.gz')
        results_file = pathlib.Path(f"task1_primal/results/2_load_balancing.csv")
    elif args.problem == 'anonymous':
        instance_files = pathlib.Path.cwd().glob('instances/3_anonymous/test/*.mps.gz')
        results_file = pathlib.Path(f"task1_primal/results/3_anonymous.csv")

    # set up the results CSV file
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_fieldnames = ['instance', 'seed', 'primal_bound_offset', 'initial_primal_bound', 'primal_integral']
    with open(results_file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
        writer.writeheader()

    # set to True to print debug information
    debug = False

    # environment
    time_limit = 10  # 5*60
    primal_bound_offset = None  # instance-specific
    initial_primal_bound = None  # instance-specific

    env = environment.RootPrimalSearch(
        time_limit=time_limit,
        observation_function=agent.ObservationFunction(problem=args.problem),
        reward_function=-environment.TimeLimitPrimalIntegral(  # minimize the primal integral <=> negated reward
            offset=lambda: primal_bound_offset,  # trick to set this value dynamically for each instance
            initial_primal_bound=lambda: initial_primal_bound,  # trick to set this value dynamically for each instance
        ),
    )

    # agent
    policy = agent.Policy(problem=args.problem)

    # evaluation loop
    for seed, instance in enumerate(instance_files):

        # seed both the agent and the environment (deterministic behavior)
        policy.seed(seed)
        env.seed(seed)

        # read the instance's initial primal and dual bounds from JSON file
        with open(instance.with_name(instance.stem).with_suffix('.json')) as f:
            instance_info = json.load(f)

        # set up the primal integral computation for that instance (primal bound initial value and offset)
        initial_primal_bound = instance_info["primal_bound"]
        primal_bound_offset = 0

        print(f"Instance {instance}")
        print(f"  seed: {seed}")
        print(f"  initial primal bound: {initial_primal_bound}")
        print(f"  primal bound offset: {primal_bound_offset}")

        # reset the policy and the environment
        policy.reset()
        observation, action_set, reward, done, info = env.reset(str(instance))
        if debug:
            print(f"  info: {info}")
            print(f"  reward: {reward}")
            print(f"  action_set: {action_set}")

        cumulated_reward = 0  # discard initial reward

        # loop over the environment
        while not done:
            action = policy(action_set, observation)
            observation, action_set, reward, done, info = env.step(action)
            if debug:
                print(f"  action: {action}")
                print(f"  info: {info}")
                print(f"  reward: {reward}")
                print(f"  action_set: {action_set}")

            cumulated_reward += reward

        print(f"  primal integral (to be minimized): {-cumulated_reward}")

        # save instance results
        with open(results_file, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
            writer.writerow({
                'instance': str(instance),
                'seed': seed,
                'primal_bound_offset': primal_bound_offset,
                'initial_primal_bound': initial_primal_bound,
                'primal_integral': -cumulated_reward,
            })
