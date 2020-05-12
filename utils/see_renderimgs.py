import gym
import time


env = gym.make('gym_flatworld:Flatworld-v0')
env.set_goal([6,5])
s = env.reset()
total_r = 0

for i in range(120):
    a = env.ideal_action(s)
    s_before = s
    s, r, done, info = env.step(a)
    total_r += r
    #print(f"action:{a},s_t:{s_before} ,s_t+1:{s}, reward:{r}, done:{done}, info:{info}")
    #rgb = env.render(mode = "rgb_array")
    time.sleep(0.20)
    env.render(mode = "rgb_array") 
    print(i)

