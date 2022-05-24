import gym
from gym import spaces
from enum import Enum
import numpy as np

class TrenchRunnerEnv(gym.Env):
    '''Environment for Trench Runner game'''
    metadata = {'render.modes': ['human']}

    NUM_ACTIONS = 3
    SIZE_Y = 9
    SIZE_X = 9
    NUM_CHANNELS = 1

    class Action(Enum):
        none = 0
        left = 1
        right = 2

    def __init__(self):
        super(TrenchRunnerEnv, self).__init__()

        self.action_space = spaces.Discrete(len(self.Action))

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.SIZE_Y, self.SIZE_X, self.NUM_CHANNELS))

        self.state = np.zeros((self.SIZE_X, self.SIZE_Y))

        self.ship_y = 1
        self.ship_x = self.SIZE_X // 2
        self.done = False

    def get_observation(self):
        observation = self.state.copy()
        observation[self.ship_y][self.ship_x] = 1
        return observation

    def step(self, action):
        if self.done:
            raise RuntimeError(
                "You are calling `step()` even though this environment has already "
                "returned done = True. You must call `reset()` before calling "
                "`step()` again.")

        action = self.Action(action)

        if action == self.Action.left:
            if self.ship_x > 0:
                self.ship_x -= 1

        elif action == self.Action.right:
            if self.ship_x < self.SIZE_X - 1:
                self.ship_x += 1

        # Shift the state down one, since the ship is moving forward
        new_state = np.concatenate((
            np.zeros((1, self.SIZE_X)),
            self.state[0:-1]
        ))

        # Choose a random position on the new row to place an obstacle
        # TODO: This can generate dead ends that are impossible to get through.
        # Need to fix that.
        obstacle_x = np.random.randint(0, self.SIZE_X)
        new_state[0][obstacle_x] = 1
        self.state = new_state

        observation = self.get_observation()
        reward = 1.0
        info = {}

        if self.state[self.ship_y][self.ship_x]:
            self.done = True

        return observation, reward, self.done, info

    def reset(self):
        self.state = np.zeros((self.SIZE_X, self.SIZE_Y))

        self.ship_y = self.SIZE_Y - 2
        self.ship_x = self.SIZE_X // 2

        self.done = False

        observation = self.get_observation()

        return observation

if __name__ == '__main__':
    env = TrenchRunnerEnv()
    observation = env.reset()
    print(observation)
    done = False

    while not done:
        observation, reward, done, info = env.step(env.Action.none)
        print(observation)

