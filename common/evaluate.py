import argparse
import csv
import json
import pathlib

import ecole as ec
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'task',
        help='Task to evaluate.',
        choices=['primal', 'dual', 'config'],
    )
    parser.add_argument(
        'problem',
        help='Problem benchmark to process.',
        choices=['item_placement', 'load_balancing', 'anonymous'],
    )
    parser.add_argument(
        '-t', '--timelimit',
        help='Episode time limit (in seconds).',
        default=argparse.SUPPRESS,
        type=float,
    )
    parser.add_argument(
        '-d', '--debug',
        help='Print debug traces.',
        action='store_true',
    )
    parser.add_argument(
        '-f', '--folder',
        help='Instance folder to evaluate.',
        default="valid",
        type=str,
        choices=("valid", "test"),
    )
    args = parser.parse_args()

    # check the Ecole version installed
    assert ec.__version__ == "0.7.3", "Wrong Ecole version."

    print(f"Evaluating the {args.task} task agent on the {args.problem} problem.")

    # collect the instance files
    if args.problem == 'item_placement':
        instances_path = pathlib.Path(f"../../instances/1_item_placement/{args.folder}/")
        results_file = pathlib.Path(f"results/{args.task}/1_item_placement.csv")
    elif args.problem == 'load_balancing':
        instances_path = pathlib.Path(f"../../instances/2_load_balancing/{args.folder}/")
        results_file = pathlib.Path(f"results/{args.task}/2_load_balancing.csv")
    elif args.problem == 'anonymous':
        instances_path = pathlib.Path(f"../../instances/3_anonymous/{args.folder}/")
        results_file = pathlib.Path(f"results/{args.task}/3_anonymous.csv")

    print(f"Processing instances from {instances_path.resolve()}")
    instance_files = list(instances_path.glob('*.mps.gz'))

    if args.problem == 'anonymous': 
        # special case: evaluate the anonymous instances five times with different seeds
        instance_files = instance_files * 5

    print(f"Saving results to {results_file.resolve()}")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_fieldnames = ['instance', 'seed', 'initial_primal_bound', 'initial_dual_bound', 'objective_offset', 'cumulated_reward']
    with open(results_file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
        writer.writeheader()

    import sys
    sys.path.insert(1, str(pathlib.Path.cwd()))

    # set up the proper agent, environment and goal for the task
    if args.task == "primal":
        from agents.primal import Policy, ObservationFunction
        from environments import RootPrimalSearch as Environment
        from rewards import TimeLimitPrimalIntegral as BoundIntegral
        time_limit = 5*60

    elif args.task == "dual":
        from agents.dual import Policy, ObservationFunction
        from environments import Branching as Environment
        from rewards import TimeLimitDualIntegral as BoundIntegral
        time_limit = 15*60

    elif args.task == "config":
        from agents.config import Policy, ObservationFunction
        from environments import Configuring as Environment
        from rewards import TimeLimitPrimalDualIntegral as BoundIntegral
        time_limit = 15*60

    # override from command-line argument if provided
    time_limit = getattr(args, "timelimit", time_limit)

    # evaluation loop
    for seed, instance in enumerate(instance_files):

        observation_function = ObservationFunction(problem=args.problem)
        policy = Policy(problem=args.problem)

        integral_function = BoundIntegral()

        env = Environment(
            time_limit=time_limit,
            observation_function=observation_function,
            reward_function=-integral_function  # negated integral (minimization)
        )

        # seed both the agent and the environment (deterministic behavior)
        observation_function.seed(seed)
        policy.seed(seed)
        env.seed(seed)

        # read the instance's initial primal and dual bounds from JSON file
        with open(instance.with_name(instance.stem).with_suffix('.json')) as f:
            instance_info = json.load(f)

        # set up the reward function parameters for that instance
        initial_primal_bound = instance_info["primal_bound"]
        initial_dual_bound = instance_info["dual_bound"]
        objective_offset = 0

        integral_function.set_parameters(
                initial_primal_bound=initial_primal_bound,
                initial_dual_bound=initial_dual_bound,
                objective_offset=objective_offset)

        print()
        print(f"Instance {instance.name}")
        print(f"  seed: {seed}")
        print(f"  initial primal bound: {initial_primal_bound}")
        print(f"  initial dual bound: {initial_dual_bound}")
        print(f"  objective offset: {objective_offset}")

        # reset the environment
        observation, action_set, reward, done, info = env.reset(str(instance), objective_limit=initial_primal_bound)

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

        print(f"  cumulated reward (to be maximized): {cumulated_reward}")

        # save instance results
        with open(results_file, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=results_fieldnames)
            writer.writerow({
                'instance': str(instance),
                'seed': seed,
                'initial_primal_bound': initial_primal_bound,
                'initial_dual_bound': initial_dual_bound,
                'objective_offset': objective_offset,
                'cumulated_reward': cumulated_reward,
            })
