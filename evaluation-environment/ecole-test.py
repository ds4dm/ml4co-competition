import ecole

env = ecole.environment.Branching(
    reward_function=-1.5 * ecole.reward.LpIterations() ** 2,
    observation_function=ecole.observation.NodeBipartite(),
)
instances = ecole.instance.SetCoverGenerator(n_rows=100, n_cols=200)

for _ in range(10):
    observation, action_set, reward_offset, done, info = env.reset(next(instances))
    while not done:
        observation, action_set, reward, done, info = env.step(action_set[0])

print("Finished")
