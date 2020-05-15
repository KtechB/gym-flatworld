from gym.envs.registration import register

register(
    id='Flatworld-v0',
    entry_point='gym_flatworld.envs:FlatworldLargeEnv',
    max_episode_steps=100,


)
register(
    id='Flatworld-v1',
    entry_point='gym_flatworld.envs:FlatworldLargeEnv',
    max_episode_steps=50,
)

register(
    id='Flatworld-v2',
    entry_point='gym_flatworld.envs:FlatworldEnv',
    max_episode_steps=50)

register(
    id='Flatworld-v3',
    entry_point='gym_flatworld.envs:FlatworldEnv',
    max_episode_steps=70)