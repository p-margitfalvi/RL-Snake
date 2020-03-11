import torch
from tqdm import tqdm

def REINFORCE(tag, env, policy, optimiser, logger=None, epochs=100, episodes=30, use_baseline=False, use_causality=False):
        # TODO: Allow for both causality and baseline
        assert not (use_baseline and use_causality)
        policy.train()
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

                    h = None

                    while not done:
                        state = torch.tensor(state, dtype=torch.double, device=self.device).unsqueeze(0)

                        action_distribution, h = policy(state, h) # distribution over actions
                        action = torch.distributions.Categorical(probs=action_distribution).sample() # sample from the distribution
                        action = int(action)

                        state, reward, done, info = env.step(action)
                        rewards.append(reward)
                        log_policy.append(torch.log(action_distribution[0, 0, action]))

                        step += 1

                        if step > 10000000:
                            print("Max step count reached, breaking.\n")
                            break

                    avg_reward += (sum(rewards) - avg_reward) / (episode + 1) # calculate moving average
                    if logger is not None:
                        logger.add_scalar(f'{tag}/Reward/Train', avg_reward, epoch*episodes + episode) # plot the latest reward

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
                optimiser.step()
                optimiser.zero_grad()

                #self.test(3)
                #self.policy.train()

        except KeyboardInterrupt:
            env.close()

        env.close()
        checkpoint = {
            'model' : policy,
            'state_dict' : policy.state_dict()
        }
        torch.save(checkpoint, f'agents/trained-agent-{tag}.pt') # save the model for later use
