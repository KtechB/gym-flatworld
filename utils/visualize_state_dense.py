import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
SIZE = 1000
def gaussian(x,mu, var):
    y = np.exp(-np.power(x- mu, 2).sum(axis=1)/(2 * var) )/np.sqrt(2* np.pi * var)
    return y
def plot_gaussian(heatmap,state, var=1):
    one_point = np.arange(0, SIZE**2).reshape(SIZE, SIZE)
    def get_dense(index_num):
        x = np.array([index_num//SIZE, index_num%SIZE])
        
def main():
    env = gym.make("flatworld")
    state_high = env.observation_space.high 
    state_low = env.observation_space.low 
    scale = SIZE / (state_high- state_low)

    def state_to_pixel(state):
        return scale * (state - state_low)
    
    states = []
    for epi in epis_list:
        stetes += epi["obs"] 

def plot_by_seaborn(path):
    with open(path, "rb") as f:
        epis = pickle.load(f)
    obs = [0] * len(epis)
    for i in range(len(epis)):
        obs[i] = epis[i]["obs"]
    obs = np.reshape(np.array(obs), (-1, epis[i]["obs"][0].shape[0]))
    sns.jointplot(obs[:,0], obs[:,1],kind="kde",xlim=[-10,10],ylim=[-10,10])
    #sns.jointplot("x", "y", obs, kind="kde")
    plt.show()
if __name__ == "__main__":
    EPIS_PATH = "data/expert_gym_flatworld:Flatworld-v0_10_s52.pkl"
    plot_by_seaborn(EPIS_PATH)

