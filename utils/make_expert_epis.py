import numpy as np
import os
import pickle
import gym


def sample_one_epi(env, pol, seed):
    env.seed(seed)
    o = env.reset()
    obs = []
    acs = []
    rews = []
    dones = []
    a_is = [{}]
    e_is = [{}]
    length = 0
    done = False
    while not(done):
        a = pol(o)
        next_o, r, done, info = env.step(a)
        obs.append(o)
        acs.append(a)
        rews.append(r)
        dones.append(done)
        o = next_o
        length += 1

    epi = dict(
        obs=np.array(obs, dtype='float32'),
        acs=np.array(acs, dtype='float32'),
        rews=np.array(rews, dtype='float32'),
        dones=np.array(dones, dtype='float32'),
        a_is=dict([(key, np.array([a_i[key] for a_i in a_is], dtype='float32'))  # action infomation
                   for key in a_is[0].keys()]),
        e_is=dict([(key, np.array([e_i[key] for e_i in e_is], dtype='float32'))  # env information
                   for key in e_is[0].keys()])
    )
    return length, epi


def sample_epis(env, max_epis_num, pol, seed):
    """
    epis_length: step number for collect
    """
    epis = []
    epis_num = 0
    epi_length = 0

    while epis_num < max_epis_num:
        seed = seed + 10
        l, epi = sample_one_epi(env, pol, seed)
        epis.append(epi)
        epi_length += l
        epis_num += 1
        if epis_num % 10 ==0:
            print(f"\repisode:{epis_num}", end="")
    print("")

    return epis


if __name__ == "__main__":
    epis_num =10 
    seed = 52
    dir_path = "./data"
    os.makedirs(dir_path, exist_ok=True)
    env_name = "gym_flatworld:Flatworld-v0"
    save_name = f"expert_{env_name}_{epis_num}_s{seed}.pkl"

    env = gym.make(env_name)
    pol = env.ideal_action

    epis = sample_epis(env, epis_num, pol, seed)
    with open(os.path.join(dir_path, save_name), mode="wb") as f:
        pickle.dump(epis, f)
    print("Done!!")