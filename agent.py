import numpy as np
import torch
import tqdm
from policy import Policy

class Agent():

    def __init__(self, tag, learning_rate, use_gpu=False):

        self.tag = tag
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print("Using" + self.device)

        # TODO: Load layer connections from JSON hyperparameters
        self.policy = Policy(layer_connections).double()
        # TODO: Load learning_rate from JSON hyperparameters
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def train(self, env, epochs=100, episodes=30, use_baseline=False, use_causality=False):
        # TODO: Allow for both causality and baseline
        assert not use_baseline and use_causality
        baseline = 0
        try:
            for epoch in tqdm(range(epochs)):
                avg_reward = 0
                objective = 0
                for episode in range(episodes):
                    done = False
                    state = env.reset()

                    log_policy = []
                    rewards = []
                    step = 0

                    while not done:
                        state = torch.tensor(state, dtype=torch.double, device=self.device)
                        state = state.view(np.prod(state.shape)) # reshape into vector

                        action_distribution = self.policy(state) # distribution over actions
                        action = torch.distributions.Categorical(probs=action_distribution).sample() # sample from the distribution
                        action = int(action)

                        state, reward, done, info = env.step(action)
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
            env.close()

        env.close()
        checkpoint = {
            'model' : policy
            'state_dict' : policy.state_dict()
        }
        torch.save(checkpoint, f'agents/trained-agent-{self.tag}.pt') # save the model for later use

    def test(self):
        pass

