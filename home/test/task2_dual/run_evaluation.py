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
    parser.add_argument(
        '-t', '--timelimit',
        help='Episode time limit (in seconds).',
        default=15*60,
        type=float,
    )
    parser.add_argument(
        '-d', '--debug',
        help='Print debug traces.',
        action='store_true',
    )
    args = parser.parse_args()

    if args.problem == 'item_placement':
        instance_files = pathlib.Path.cwd().glob('instances/1_item_placement/test/*.mps.gz')
        results_file = pathlib.Path(f"task2_dual/results/1_item_placement.csv")
    elif args.problem == 'load_balancing':
        instance_files = pathlib.Path.cwd().glob('instances/2_load_balancing/test/*.mps.gz')
        results_file = pathlib.Path(f"task2_dual/results/2_load_balancing.csv")
    elif args.problem == 'anonymous':
        instance_files = pathlib.Path.cwd().glob('instances/3_anonymous/test/*.mps.gz')
        results_file = pathlib.Path(f"task2_dual/results/3_anonymous.csv")

    # set up the results CSV file
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_fieldnames = ['instance', 'seed', 'dual_bound_offset', 'initial_dual_bound', 'dual_integral']
    with open(results_file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
        writer.writeheader()

    # environment
    dual_bound_offset = None  # instance-specific
    initial_dual_bound = None  # instance-specific

    env = environment.Branching(
        time_limit=args.timelimit,
        observation_function=agent.ObservationFunction(problem=args.problem),
        reward_function=-environment.TimeLimitDualIntegral(  # minimize the dual integral <=> negated reward
            offset=lambda: dual_bound_offset,  # trick to set this value dynamically for each instance
            initial_dual_bound=lambda: initial_dual_bound,  # trick to set this value dynamically for each instance
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

        # set up the dual integral computation for that instance (dual bound initial value and offset)
        initial_dual_bound = instance_info["dual_bound"]
        dual_bound_offset = 0
        objective_limit = instance_info["primal_bound"]

        print(f"Instance {instance}")
        print(f"  seed: {seed}")
        print(f"  initial dual bound: {initial_dual_bound}")
        print(f"  dual bound offset: {dual_bound_offset}")
        print(f"  objective_limit: {objective_limit}")

        # reset the policy and the environment
        policy.reset()
        observation, action_set, reward, done, info = env.reset(str(instance), objective_limit=objective_limit)
        if args.debug:
            print(f"  info: {info}")
            print(f"  reward: {reward}")
            print(f"  action_set: {action_set}")

        cumulated_reward = 0  # discard initial reward

        # loop over the environment
        while not done:
            action = policy(action_set, observation)
            observation, action_set, reward, done, info = env.step(action)
            if args.debug:
                print(f"  action: {action}")
                print(f"  info: {info}")
                print(f"  reward: {reward}")
                print(f"  action_set: {action_set}")

            cumulated_reward += reward

        print(f"  dual integral (to be minimized): {-cumulated_reward}")

        # save instance results
        with open(results_file, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
            writer.writerow({
                'instance': str(instance),
                'seed': seed,
                'dual_bound_offset': dual_bound_offset,
                'initial_dual_bound': initial_dual_bound,
                'dual_integral': -cumulated_reward,
            })
