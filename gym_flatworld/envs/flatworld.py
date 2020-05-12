import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import time

X = 1
Y = 1
SPEED_SCALE = 0.5 * 0.1
GOAL = np.array([0, 0])
Terminal = False
eps = X * 1e-5


class FlatworldEnv(gym.Env):
    metadata = {'render.modes': ['human'],
                'video.frames_per_second': 30}

    def __init__(self, seed=0):
        self.GOAL = np.array([0, 0])  # set goal state (0, 0)

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

    def set_goal(self, goal):
        assert self.observation_space.contains(
            goal), "goal must be np.array with shape (2,)"
        self.GOAL = goal

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
                 diagnosic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        next_state = self._get_next_state(self.state, action)
        reward = self._get_reward(self.state, action, next_state)
        
        if Terminal:
            if self._is_terminal_state(next_state):
                done = True
            else:
                done = False
        else:
            done = False

        self.state = next_state
        return self.state, reward, done, {"goal": self.GOAL}

    def _is_terminal_state(self, state):
        if np.abs(state).all() < eps:
            return True
        else:
            return False

    def _get_next_state(self, state, action):
        norm = np.linalg.norm(action)
        if norm > 1:
            move_dist = SPEED_SCALE*(action/norm)  # limit max speed =1
        else:
            move_dist = SPEED_SCALE * action
        next_state = np.clip(
            state + move_dist, self.observation_space.low, self.observation_space.high)
        return next_state

    def _get_reward(self, state, action, next_state):
        lower_bound = -10
        reward = - np.square(next_state - self.GOAL).sum() / X
        reward = max(reward, lower_bound)
        return reward

    def reset(self):
        self.state = self.init_state()
        return self.state

    def set_init_state(self, s):
        assert self.observation_space.contains(s)
        self.state = s
        return self.state

    def render(self, mode='human', **kwargs):
        screen_width = 600
        screen_height = 600

        world_width = 2 * X
        scale = screen_width/world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            xs = np.linspace(-X, X, 100) - X
            ys = np.linspace(-Y, Y, 100) - Y - 5
            x_scaled = X * scale
            y_scaled = Y * scale
            ones_x = np.ones_like(xs)
            ones_y = np.ones_like(ys)
            lines = [list(zip((xs)*scale, 2*Y*scale * ones_x)),
                     list(zip((xs)*scale, 0 * ones_x)),
                     list(zip(2*X*scale * ones_y, (ys)*scale)),
                     list(zip(0 * ones_y, (ys)*scale))]
            for line in lines:
                xys = line
                track = rendering.make_polyline(xys)
                track.set_linewidth(4)
                track.set_color(0, .5, 0)
                self.viewer.add_geom(track)
                # clearance = 10
            agent_size = 5
            agent = rendering.make_circle(agent_size, filled=True)
            self.agenttrans = rendering.Transform(
                translation=(0, 0))  # translation = (dx,dy)ずらす
            agent.add_attr(self.agenttrans)
            agent.set_color(.9, .5, .5)
            self.viewer.add_geom(agent)
            horizontal_line = rendering.Line(
                (0, y_scaled), (2*x_scaled, y_scaled))
            self.viewer.add_geom(horizontal_line)
            for i in range(20):
                
                pos_x = i * (screen_width/20)
                line_length = 3 if i % 2 == 0 else 1

                flagpole = rendering.Line(
                    (pos_x, y_scaled-line_length), (pos_x, y_scaled + line_length))

                self.viewer.add_geom(flagpole)

            flag_size = 7

            flag = rendering.make_circle(flag_size, filled=True)
            self.flagtrans = rendering.Transform(
                translation=(0, 0))  # translation = (dx,dy)ずらす
            flag.add_attr(self.flagtrans)
            flag.set_color(.5, .9, .9)
            self.viewer.add_geom(flag)
            """
            self.flagtrans = rendering.Transform(
                )
            flagx = (self.GOAL[0]+X)*scale
            print(self.GOAL)
            flagy1 = (self.GOAL[1] + Y) * scale
            flagy2 = flagy1 + 20
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            flagpole.add_attr(rendering.Transform(translation=(0, 0.1)))
            flagpole.add_attr(self.flagtrans)

            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2-8), (flagx+20, flagy2-4)])
            flag.set_color(.8, .8, 0)

            flag.add_attr(self.flagtrans)
            self.viewer.add_geom(flag)
            """
        pos = self.state
        self.flagtrans.set_translation(
            (self.GOAL[0]+X)*scale, (self.GOAL[1] + Y) * scale
        )
        self.agenttrans.set_translation(
            (pos[0] + X) * scale, (pos[1] + Y) * scale)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def ideal_action(self, state=None):
        if state is None:
            state = self.state

        from_goal = self.state - self.GOAL
        dist = np.linalg.norm(from_goal)
        if dist < SPEED_SCALE:
            a = -from_goal/SPEED_SCALE
        else:
            a = -from_goal / (dist * SPEED_SCALE)

        return a

    def close(self):
        pass


def test_run():
    env = FlatworldEnv(seed=42)
    s = env.reset()
    total_r = 0
    for i in range(100):
        a = env.ideal_action(s)
        s_before = s
        s, r, done, info = env.step(a)
        total_r += r
    return True


if __name__ == "__main__":
    env = FlatworldEnv(seed=42)
    env.set_goal([1, 5])
    s = env.reset()
    total_r = 0

    for i in range(100):
        a = env.ideal_action(s)
        s_before = s
        s, r, done, info = env.step(a)
        total_r += r
        print(
            f"action:{a},s_t:{s_before} ,s_t+1:{s}, reward:{r}, done:{done}, info:{info}")
        # rgb = env.render(mode = "rgb_array")
