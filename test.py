from SnakeV0 import Snake
from time import sleep
from model import SimpleA2C
from algorithms import A2CAgent

import torch
import wandb
import numpy as np

def main_auto():
    env = Snake()
    env.reset(snake_size=5)
    while True:
        for i in range(0, 4):
            for _ in range(0, 3):
                state, reward, done, info = env.step(i)
                env.render()
                sleep(0.2)

    env.close()

def main_manual():
    env = Snake(6, 6)
    env.reset(snake_size=2)
    while True:
        key = input('input')
        if key == 'd':
            action = 1
        elif key == 'w':
            action = 0
        elif key == 's':
            action = 2
        else:
            action = 3

        state, reward, done, info = env.step(action)
        print(state)
        if done:
            env.close()
            break
        env.render()

    env.close()

def agent_test():
    env = Snake(6, 6)
    env.reset()
    env.render()
    input("press any key to run the test")
    agent = A2CAgent.load('agents/agents-checkpoint-step-4000.pt', env)
    agent.test(10)

def agent_train():
    env = Snake(6, 6)
    input_shape = np.prod(env.observation_space.sample().shape)
    model = SimpleA2C(input_shape, 4).double()
    wandb.init(project='rl-snake')
    agent = A2CAgent(env, model, torch.optim.Adam(model.parameters(), lr=5e-4),
            entropy_coeff=0.1, critic_coeff=1, batch_size_ep=10, n_optim=5, device='cpu', discount=0.95, logger=wandb)

    agent.train(8000, save_freq=1000, keep_num_chckpt=4)
    agent.test(10)

if __name__ == '__main__':
    #agent_train()
    #main_manual()
    agent_test()
