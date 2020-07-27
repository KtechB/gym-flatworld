# 2D continuous environment
Simple 2D continuous environment for RL(Developping)

After you have installed your package with 

```
pip install -e .
```
you can create an instance of the environment with 
```
gym.make('gym_flatworld:Flatworld-v0')
```

# version
Flatworld-v0:  max_episode_steps:100 scale: (-10,10)
start at any state 
Flatworld-v1:  max_episode_steps:50 scale: (-10,10)
start at any state 
Flatworld-v2: max_episode_steps:50 scale:(-1,1)
start at y = -0.5 x = (-1,1)

Flatworld-v3: max_episode_steps:70 scale:(-1,1)
start at y = ±0.8 x = (-1,1)


FlatworldMultiGoal