import json
import numpy as np
import torch
from tqdm import tqdm
from time import sleep
import policy
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
        agent = cls(tag, env, use_gpu, log)

        f = open(hyperparam_path)
        hyperparams = json.load(f)

        agent.learning_rate = hyperparams['learning_rate']

        connection_mode = hyperparams['connection_mode']
        if hyperparams['architecture'] == 'FNN':
            layer_connections = hyperparams['hidden_layers']
            layer_connections = np.multiply(layer_connections, int(np.prod(env.observation_space.shape))) if connection_mode == "multiplicative" else layer_connections
            layer_connections = np.insert(layer_connections, 0, int(np.prod(env.observation_space.shape)))
            layer_connections = np.append(layer_connections, int(env.action_space.n))
            agent.policy = policy.FNNPolicy(layer_connections, output_distribution=True).to(dtype=torch.double,
                                                                                     device=agent.device)
        else:
            cnn_dict = hyperparams['CNN_hidden_layers']
            fnn_layers = hyperparams['FNN_hidden_layers']
            # Assumes square observation space
            input_size = cnn_dict['channels'][-1]*policy.__convolution_output_size__(env.observation_space.shape[-1][-1], cnn_dict['kernel_sizes'], cnn_dict['strides'])**2
            if connection_mode == "multiplicative":
                fnn_layers = np.multiply(fnn_layers, input_size)
            fnn_layers = np.insert(fnn_layers, 0, input_size)
            fnn_layers = np.append(fnn_layers, int(env.action_space.n))
            agent.policy = policy.CNNPolicy(cnn_dict, fnn_layers.astype(int), output_distribution=True).to(dtype=torch.double, device=agent.device)

        agent.optimiser = torch.optim.Adam(agent.policy.parameters(), lr=agent.learning_rate)
        return agent

    # Loads model from checkpoint
    @classmethod
    def for_inference(cls, tag, env, model_path, use_gpu=False):
        agent = cls(tag, env, use_gpu)
        if not agent.device == "cuda":
            checkpoint = torch.load(model_path, map_location="cpu")
            agent.policy = checkpoint['model']
            agent.policy.load_state_dict(checkpoint['state_dict'])
        else:
            agent.policy = torch.load(model_path)
        return agent


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
                        state = torch.tensor(state, dtype=torch.double, device=self.device).unsqueeze(0)

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
            action = torch.argmax(action_distribution).item()
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
        gr_policy = self.__create_greedy_policy__(self.policy)

        for episode in range(episodes):
            done = False
            state = self.env.reset()

            while not done:
                state = torch.tensor(state, dtype=torch.double, device=self.device)
                state = state.view(np.prod(state.shape))
                action = gr_policy(state)
                state, reward, done, info = self.env.step(action)
                self.env.render()
                sleep(0.2)



