import json
import numpy as np
import torch
from tqdm import tqdm
from time import sleep
from policy import FNNPolicy
from torch.utils.tensorboard import SummaryWriter

# TODO: Implement an actual RNN for memory
class Agent():

    def __init__(self, tag, env, use_gpu=False, log=False):
        self.tag = tag
        self.env = env

        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print("Using " + self.device + "\n")

        if log:
            self.writer = SummaryWriter()

    # Creates new model for training
    @classmethod
    def for_training(cls, tag, hyperparam_path, env, log=False, use_gpu=False):
        cls.__init__(tag, env, use_gpu, log)

        f = open(hyperparam_path)
        hyperparams = json.load(f)

        cls.learning_rate = hyperparams['learning_rate']

        connection_mode = hyperparams['connection_mode']
        layer_connections = hyperparams['hidden_layers']
        # Account for the extra "memory" state along dimension 1
        layer_connections.insert(0, int(np.prod(env.observation_space.shape)))
        layer_connections.append(int(env.action_space.n))

        # Multiplicative mode means layer_connections contain the multiplier of first layer nodes in that layer
        if connection_mode == "multiplicative":
            for idx in range(1, len(layer_connections) - 1):
                layer_connections[idx] *= layer_connections[0]

        cls.policy = FNNPolicy(layer_connections, output_distribution=True).to(dtype=torch.double, device=cls.device)
        cls.optimiser = torch.optim.Adam(cls.policy.parameters(), lr=cls.learning_rate)

    # Loads model for inference
    @classmethod
    def for_inference(cls, tag, env, model_path, log=False, use_gpu=False):
        cls.__init__(tag, env, use_gpu, log)
        cls.policy = torch.load(model_path)


    def train(self, epochs=100, episodes=30, use_baseline=False, use_causality=False):
        # TODO: Allow for both causality and baseline
        assert not (use_baseline and use_causality)
        self.policy.train()
        baseline = 0
        try:
            for epoch in tqdm(range(epochs)):
                avg_reward = 0
                objective = 0
                for episode in range(episodes):
                    done = False
                    state = self.env.reset()

                    log_policy = []
                    rewards = []
                    step = 0

                    h = None

                    while not done:
                        state = torch.tensor(state, dtype=torch.double, device=self.device)
                        state = state.view(1, 1, np.prod(state.shape)) # reshape into vector

                        action_distribution, h = self.policy(state, h) # distribution over actions
                        action = torch.distributions.Categorical(probs=action_distribution).sample() # sample from the distribution
                        action = int(action)

                        state, reward, done, info = self.env.step(action)
                        rewards.append(reward)
                        log_policy.append(torch.log(action_distribution[0, 0, action]))

                        step += 1

                        if step > 10000000:
                            print("Max step count reached, breaking.\n")
                            break

                    avg_reward += (sum(rewards) - avg_reward) / (episode + 1) # calculate moving average
                    if self.writer is not None:
                        self.writer.add_scalar(f'{self.tag}/Reward/Train', avg_reward, epoch*episodes + episode) # plot the latest reward

                    if use_baseline:
                        baseline += (sum(rewards) - baseline) / (epoch*episodes + episode + 1)

                    for idx in range(step):
                        if use_causality:
                            weight = sum(rewards[idx:])
                        else:
                            weight = sum(rewards) - baseline
                        objective += weight*log_policy[idx]

                objective /= episodes # average over episodes
                objective *= -1 # minimising this means maximising rewards

                # Policy update
                objective.backward()
                self.optimiser.step()
                self.optimiser.zero_grad()

        except KeyboardInterrupt:
            self.env.close()

        self.env.close()
        checkpoint = {
            'model' : self.policy,
            'state_dict' : self.policy.state_dict()
        }
        torch.save(checkpoint, f'agents/trained-agent-{self.tag}.pt') # save the model for later use

    def __create_greedy_policy__(self, behaviour_func):
        def policy(observation):
            action_distribution = behaviour_func(observation)
            action = np.argmax(action_distribution)
            return action
        return policy

    def __create_stochastic_policy__(self, behaviour_func):
        def policy(observation):
            action_distribution = behaviour_func(observation)
            action = torch.distributions.Categorical(probs= action_distribution).sample().item()
            return action
        return policy

    def test(self, episodes=10):
        self.policy.eval()
        policy = self.__create_greedy_policy__(self.policy)
        for episode in range(episodes):
            done = False
            state = self.env.reset()

            while not done:
                state = torch.tensor(state, dtype=torch.double, device=self.device)
                state = state.view(np.prod(state.shape))
                action = policy(state)
                state, reward, done, info = self.env.step(action)
                self.env.render()
                sleep(0.2)



