from gym.envs.registration import register

register(
    id='Flatworld-v0',
    entry_point='gym_flatworld.envs:FlatworldEnv',
)
