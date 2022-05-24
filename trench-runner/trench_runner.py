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
        self.screen = None
        self.clock = None

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

    def render(self, mode='human'):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install gym`')

        screen_scale = 75

        screen_width = self.SIZE_X * screen_scale
        screen_height = self.SIZE_Y * screen_scale

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        # Draw the ship
        gfxdraw.box(
            self.surf,
            pygame.Rect(
                (self.ship_x * screen_scale, self.ship_y * screen_scale),
                (screen_scale, screen_scale)),
            (90, 100, 255))

        # Draw the obstacles. If there's a collision, the obstacle will cover
        # the ship, as if the ship was destroyed
        for state_y in range(0, self.state.shape[0]):
            for state_x in range(0, self.state.shape[1]):
                if self.state[state_y][state_x]:
                    gfxdraw.box(
                        self.surf,
                        pygame.Rect(
                            (state_x * screen_scale, state_y * screen_scale),
                            (screen_scale, screen_scale)),
                        (255, 100, 100))

        self.screen.blit(self.surf, (0, 0))

        if mode == 'human':
            self.clock.tick(24)

        pygame.event.pump()
        pygame.display.flip()

    def check_quit(self):
        if self.screen is not None:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True

        return False


    def close(self, force=False):
        if self.screen is not None:
            import pygame

            if not force:
                while not self.check_quit():
                    pass

            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    env = TrenchRunnerEnv()
    observation = env.reset()
    env.render()

    done = False

    while True:
        observation, reward, done, info = env.step(env.Action.left)
        env.render()

        if done:
            done = False
            observation = env.reset()
            env.render()

        if env.check_quit():
            env.close(force=True)
