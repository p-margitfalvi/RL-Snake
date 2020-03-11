import torch
import numpy as np
from tqdm import tqdm

def REINFORCE(tag, env, policy, optimiser, device, logger=None, epochs=100, episodes=30, use_baseline=False, use_causality=False):
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
                        state = torch.tensor(state, dtype=torch.double, device=device).unsqueeze(0)

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

def A2C(tag, env, actor_critic, optimiser, gamma, logger=None, epochs=100):
    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for epoch in range(epochs):
        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        done = False
        steps = 0
        h_a, h_c = None, None
        while not done:
            (policy_dist, h_a), (value, h_c) = actor_critic.forward(state, h_a, h_c)
            value = value.detach().numpy()[0, 0]

            action = torch.distributions.Categorical(provs=policy_dist).sample().item()
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            entropy = -torch.sum(torch.mean(policy_dist) * np.log(policy_dist))
            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy

            if done or steps == 1000000:
                Qval, _ = actor_critic.forward(state)
                Qval = Qval.detach().numpy()[0, 0]
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-10:]))
                if epoch % 10 == 0:
                    print(
                        "episode: {}, reward: {}, total length: {}, average length: {} \n".format(epoch,
                                                                                                  np.sum(rewards),
                                                                                                  steps,
                                                                                                  average_lengths[
                                                                                                      -1]))
                break
            steps += 1
        if logger is not None:
            logger.add_scalar(f'{tag}/Reward/Train', np.mean(rewards), epoch)  # plot the latest reward
        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + gamma * Qval
            Qvals[t] = Qval

        # update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5*advantage.pow(2).mean()
        loss = actor_loss + critic_loss + 0.001 * entropy_term

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
