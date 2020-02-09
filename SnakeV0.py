import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Snake(gym.Env):

    def __init__(self, size=16):
        super(Snake, self).__init__()

        # 0 -> left
        # 1 -> up
        # 2 -> right
        # 3 -> down
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.MultiBinary(3)

        self.size = size

    def step(self, action):
        assert self.action_space.contains(action), 'Invalid action'

        reward = -0.1

        action_vector = np.zeros(2)
        if action == 0:
            action_vector[0] = -1
        elif action == 1:
            action_vector[1] = 1
        elif action == 2:
            action_vector[0] = 1
        elif action == 3:
            action_vector[1] = -1

        # Flag indicating whether snake ate this step
        ate = False
        tail_pos = self.snake[-1]
        for idx in range(self.snake.shape[0] - 1, -1, -1):

            if idx == 0:

                # Creates new position for the head by adding the movement direction to the current head position
                new_head_pos = np.add(self.snake[0], action_vector)
                if np.array_equal(new_head_pos, self.apple):
                    reward = 10
                    self.apple_spawn_counter = self.rng.randint(3, 7)
                    self.apple = None
                    ate = True

                # Collision detection
                for j in range(1, self.snake.shape[0]):
                    if np.array_equal(new_head_pos, self.snake[j]):
                        reward -= 100
                        # TODO: Add point of collision for info
                        return self.get_state(), reward, True, {}

                self.snake[0] = new_head_pos

            # Update the body positions to their next positions from the back
            else:
                self.snake[idx] = self.snake[idx - 1]

        # If there's no apple in the world, step decrements the spawn counter
        if self.apple is None:
            self.apple_spawn_counter -= 1

        if self.apple_spawn_counter == 0:
            # Generate apple coordinates
            self.apple = self.generate_apple()

        if ate:
            self.snake = np.append(self.snake, tail_pos)


        return self.get_state(), reward, False, {}

    def render(self, values=None, mode='human'):
        screen_width = 600
        screen_height = 600

        square_size_height = screen_height / self.n_squares_height
        square_size_width = screen_width / self.n_squares_width

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):

        self.seed(seed)

        midpoint = self.size // 2

        self.snake = np.zeros([2, 2], dtype=np.int)

        self.snake[0] = [midpoint, midpoint]
        self.snake[1] = [midpoint, midpoint + 1]

        self.apple = None
        self.apple = self.generate_apple()
        self.apple_spawn_counter = self.rng.randint(3, 7)

        return self.get_state()

    # Generates an apple at one of the free spots
    def generate_apple(self):
        state = self.get_state()

        free_indices = np.argwhere(state == 0)

        free_x = free_indices[0]
        free_y = free_indices[1]

        return np.array([self.rng.choice(free_x), self.rng.choice(free_y)])


    def get_state(self):

        cur_state = np.zeros([self.size, self.size, self.size])

        for idx in range(self.snake.shape[0]):
            cur_state[0, self.snake[idx, 0], self.snake[idx, 1]] = 1

        if self.apple is not None:
            cur_state[1, self.apple[0], self.apple[1]] = 1

        return cur_state



