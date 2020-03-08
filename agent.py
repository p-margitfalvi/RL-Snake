import json
import numpy as np
import torch
import tqdm
from policy import Policy

class Agent():

    def __init__(self, tag, hyperparam_path, env, use_gpu=False):

        self.tag = tag
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print("Using " + self.device + "\n")

        f = open(hyperparam_path)
        hyperparams = json.load(f)

        self.learning_rate = hyperparams['learning_rate']

        connection_mode = hyperparams['connection_mode']
        layer_connections = hyperparams['hidden_layers']

        layer_connections.insert(0, np.prod(env.observation_space.shape))
        layer_connections.append(env.action_space.n)

        # Multiplicative mode means layer_connections contain the multiplier of first layer nodes in that layer
        if connection_mode == "multiplicative":
            for idx in range(1, len(layer_connections) - 1):
                layer_connections[idx] *= layer_connections[0]

        self.policy = Policy(layer_connections, connection_mode).double()
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def train(self, epochs=100, episodes=30, use_baseline=False, use_causality=False):
        # TODO: Allow for both causality and baseline
        assert not use_baseline and use_causality
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

                    while not done:
                        state = torch.tensor(state, dtype=torch.double, device=self.device)
                        state = state.view(np.prod(state.shape)) # reshape into vector

                        action_distribution = self.policy(state) # distribution over actions
                        action = torch.distributions.Categorical(probs=action_distribution).sample() # sample from the distribution
                        action = int(action)

                        state, reward, done, info = self.env.step(action)
                        rewards.append(reward)
                        log_policy.append(torch.log(action_distribution[action]))

                        step += 1

                    avg_reward += (sum(reward) - avg_reward) / (episode + 1) # calculate moving average

                    self.writer.add_scalar(f'{self.tag}/Reward/Train', avg_reward, epoch*episodes + episode) # plot the latest reward

                    if use_baseline:
                        baseline += (sum(rewards) - baseline) / (epoch*episodes + episode)

                    for idx in range(step):
                        if use_causality:
                            weight = sum(rewards[idx:])
                        else:
                            weight = sum(rewards) - baseline
                        objective += weight*log_policy[idx]

                objective /= episodes # average over episodes
                objective *= -1 # minimising this means maximising rewards

                # Policy update
                self.optimiser.backward()
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

    def test(self):
        pass

