import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import time

class Snake(gym.Env):

    __crash_reward__ = -1
    __apple_reward__ = 1
    __movement_reward__ = -0.02

    def __init__(self, h_size=64, v_size=64):
        super(Snake, self).__init__()

        # 0 -> left
        # 1 -> up
        # 2 -> right
        # 3 -> down
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.MultiBinary((2, h_size, v_size))

        self.n_h_squares = h_size
        self.n_v_squares = v_size

        self.viewer = None
        self.ate = False

    def step(self, action):
        assert self.action_space.contains(action), 'Invalid action'

        reward = self.__movement_reward__

        action_vector = np.zeros(2)
        # 0 for left, 1 for up, 2 for right, 3 for down
        if action == 0:
            action_vector[0] = -1
        elif action == 1:
            action_vector[1] = 1
        elif action == 2:
            action_vector[0] = 1
        elif action == 3:
            action_vector[1] = -1

        # Flag indicating whether snake ate this step
        self.ate = False
        tail_pos = self.snake[-1]

        # Creates new position for the head by adding the movement direction to the current head position
        new_head_pos = np.add(self.snake[0], action_vector)

        if np.array_equal(new_head_pos, self.snake[1]):
            return self.get_state(), self.__crash_reward__, True, {}

        for idx in range(self.snake.shape[0] - 1, -1, -1):

            if idx == 0:

                # Check if snake went out of bounds
                if new_head_pos[0] >= self.n_h_squares or new_head_pos[1] >= self.n_h_squares or np.amin(new_head_pos) < 0:
                    # TODO: Add point of collision for info
                    return self.get_state(), self.__crash_reward__, True, {}

                if np.array_equal(new_head_pos, self.apple):
                    reward = self.__apple_reward__
                    self.apple_spawn_counter = 1 # self.rng.randint(20, 50)
                    self.apple = None
                    self.ate = True

                # Collision detection
                for j in range(1, self.snake.shape[0]):
                    if np.array_equal(new_head_pos, self.snake[j]):
                        # TODO: Add point of collision for info
                        return self.get_state(), self.__crash_reward__, True, {}

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
            self.apple_spawn_counter = -1

        if self.ate:
            self.snake = np.append(self.snake, np.expand_dims(tail_pos, axis=0), axis=0)

        return self.get_state(), reward, False, {}

    def render(self, values=None, mode='human'):
        screen_width = 600
        screen_height = 600

        self.square_size_height = screen_height / self.n_v_squares
        self.square_size_width = screen_width / self.n_h_squares

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.rendering = rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # the goal
            l, r, t, b = -self.square_size_width / 2, self.square_size_width / 2, self.square_size_height / 2, -self.square_size_height / 2
            self.goal = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.goal.set_color(1, 0, 0)
            self.goaltrans = rendering.Transform()
            self.goal.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal)

            l, r, t, b = -self.square_size_width / 2, self.square_size_width / 2, self.square_size_height / 2, -self.square_size_height / 2
            self.snake_body = [rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)]) for j in range(0, self.snake.shape[0])]

            self.snake_transforms = [rendering.Transform() for j in range(0, self.snake.shape[0])]

            for i in range(0, self.snake.shape[0]):
                self.snake_body[i].add_attr(self.snake_transforms[i])
                self.viewer.add_geom(self.snake_body[i])
                sq_x, sq_y = self.convert_pos_to_xy(self.snake[i], (self.square_size_width, self.square_size_height))
                self.snake_transforms[i].set_translation(sq_x, sq_y)
                self.snake_body[i].set_color(0, 0, 0)

        if self.ate:
            l, r, t, b = -self.square_size_width / 2, self.square_size_width / 2, self.square_size_height / 2, -self.square_size_height / 2
            self.snake_body.append(self.rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)]))
            self.snake_transforms.append(self.rendering.Transform())
            self.snake_body[-1].add_attr(self.snake_transforms[-1])
            self.viewer.add_geom(self.snake_body[-1])
            sq_x, sq_y = self.convert_pos_to_xy(self.snake[-1], (self.square_size_width, self.square_size_height))
            self.snake_transforms[-1].set_translation(sq_x, sq_y)
            self.snake_body[-1].set_color(0, 0, 0)


        for i in range(0, self.snake.shape[0]):
            sq_x, sq_y = self.convert_pos_to_xy(self.snake[i], (self.square_size_width, self.square_size_height))
            self.snake_transforms[i].set_translation(sq_x, sq_y)

        if self.apple is not None:
            goal_pos = self.apple
            goal_x, goal_y = self.convert_pos_to_xy(goal_pos, (self.square_size_width, self.square_size_height))
            self.goaltrans.set_translation(goal_x, goal_y)
            self.goal.set_color(1, 0, 0)
        else:
            # Set color to white
            self.goal.set_color(1, 1, 1)

        if values is not None:
            maxval, minval = values.max(), values.min()
            rng = maxval - minval
            for i, row in enumerate(values):
                for j, val in enumerate(row):
                    if rng == 0:
                        col = 1
                    else:
                        col = (maxval - val) / rng
                    self.squares[i][j].set_color(col, 1, col)
        time.sleep(0.1)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def reset(self, seed=None, snake_size=2):

        self.seed(seed)

        h_midpoint = self.n_h_squares // 2
        v_midpoint = self.n_v_squares // 2

        self.snake = np.zeros([snake_size, 2], dtype=np.int)

        for i in range(snake_size):
            self.snake[i] = [h_midpoint, v_midpoint + i]

        self.apple = None
        self.apple = self.generate_apple()
        self.apple_spawn_counter = self.rng.randint(3, 7)

        if self.viewer is not None:
            self.snake_transforms = None
            self.snake_body = None
            self.viewer.close()
            self.viewer = None

        return self.get_state()

    # Generates an apple at one of the free spots
    def generate_apple(self):
        state = self.get_state()

        # Free indices are those where there isnt snake body
        free_indices = np.argwhere(state[0] == 0)
        choice = self.rng.randint(0, free_indices.shape[0])
        return free_indices[choice]

    def convert_pos_to_xy(self, pos, size):
        x = (pos[1] + 0.5) * size[0]
        y = (self.n_v_squares - pos[0] - 0.5) * size[1]
        return x, y


    def get_state(self):

        cur_state = np.zeros([2, self.n_h_squares, self.n_v_squares])

        for idx in range(self.snake.shape[0]):
            cur_state[0, self.snake[idx, 0], self.snake[idx, 1]] = 1

        if self.apple is not None:
            cur_state[1, self.apple[0], self.apple[1]] = 1

        # Head pos in third channel
        #cur_state[2, self.snake[0, 0], self.snake[0, 1]] = 1

        return cur_state



