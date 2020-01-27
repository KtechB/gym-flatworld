import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

X = 10
Y = 10
SPEED_SCALE = 0.5
GOAL = np.array([0, 0])

def non_liner_func(x,w ,output_dim):
    """w is list of numpy array"""
    return x 
class FlatworldEnv(gym.Env):
    metadata = {'render.modes': ['human'],
                'video.frames_per_second': 30}

    def __init__(self, seed=0):

        self.state_low = np.array([-X, -Y])
        self.state_high = np.array([X, Y])
        self.action_min = np.array([-1, -1])
        self.action_max = np.array([1, 1])

        self.viewer = None

        self.action_space = spaces.Box(low=self.action_min, high=self.action_max,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high,
                                            dtype=np.float32)

        self.seed(seed)
        self.reset()

    def init_state(self):
        return np.random.uniform(low=self.state_low, high=self.state_high, size=2)

    def seed(self, seed=42):
        np.random.seed(seed)

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : Tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        next_state =self._get_next_state(self.state, action)
        reward = self._get_reward(self.state, action, next_state)
        done = False
        self.state = next_state
        return self.state, reward, done, {}


    def _get_next_state(self, state, action):
        x, y = state
        a_x, a_y = action
        norm = np.linalg.norm(action)
        if norm > 1:
            move_dist = SPEED_SCALE*action/norm  # limit max speed =1
        else:
            move_dist = SPEED_SCALE * action
        next_state = np.clip(self.state + move_dist, self.observation_space.low, self.observation_space.high) 
        return next_state 
    def _get_reward(self, state, action, next_state):
        lower_bound = -10
        reward = - np.square(next_state - GOAL).sum() / X
        reward = max(reward, lower_bound)
        return reward

    def reset(self):
        self.state = self.init_state()
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = X
        scale = screen_width/world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            xs = np.linspace(-X, X, 100) - X
            ys = np.linspace(-Y, Y, 100) - Y
            lines = [list(zip((xs)*scale, 2*Y*scale)),
                     list(zip((xs)*scale, 0)),
                     list(zip(2*X*scale, (ys)*scale)),
                     list(zip(0, (ys)*scale))]

            for line in lines:
                xys = line
                self.track = rendering.make_polyline(xys)
                self.track.set_linewidth(4)
                self.viewer.add_geom(self.track)
            #clearance = 10
            agent_size = 3
            agent = rendering.make_circle(agent_size, filled=True)
            self.agenttrans = rendering.Transform(
                translation=(0, 0))  # translation = (dx,dy)ずらす
            agent.add_attr(self.agenttrans)
            agent.set_color(.9, .5, .5)
            self.viewer.add_geom(agent)

            """
            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))#移動
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            """
            flagx = (GOAL[0]+X)*scale
            flagy1 = (GOAL[1] + Y) * scale
            flagy2 = flagy1 + 20
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2-8), (flagx+20, flagy2-4)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state
        self.agenttrans.set_translation(
            (pos[0] + X) * scale, (pos[1] + Y) * scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def ideal_action(self):
        a_x, a_y = np.clip(-self.state.copy()/SPEED_SCALE,self.action_min, self.action_max) 
        return (a_x, a_y)
        
                
if __name__ == "__main__":
    env = FlatworldEnv(seed=42)
    s = env.reset()
    total_r = 0
    for i in range(100):
        a = env.ideal_action()
        s_before = s
        s, r, done, info = env.step(a)
        total_r += r
        print(f"action:{a},s_t:{s_before} ,s_t+1:{s}, reward:{r}, Done:{done}")
        #rgb = env.render(mode = "rgb_array")
    print(f"total reward:{total_r}")
