import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import wandb

#wandb.init(project= 'rl-snake')

def REINFORCE(tag, env, policy, optimiser, device, logger=None, epochs=100, episodes=30, recurrent_model=False, use_baseline=False, use_causality=False):
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
                        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                        if recurrent_model:
                            action_distribution, h = policy(state, h) # distribution over actions
                        else:
                            action_distribution = policy(state)
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
            checkpoint = {
                'model': policy,
                'state_dict': policy.state_dict()
            }
            torch.save(checkpoint, f'agents/aborted-agent-{tag}.pt')

        env.close()
        checkpoint = {
            'model' : policy,
            'state_dict' : policy.state_dict()
        }
        torch.save(checkpoint, f'agents/trained-agent-{tag}.pt') # save the model for later use

def A2C(tag, env, actor_critic, optimiser, gamma, entropy_coeff, critic_coeff, device,
        regularize_returns=False, recurrent_model=False, logger=None, test_func=None, test_spacing= -1, epochs=100):

    actor_critic.train()
    #wandb.watch(actor_critic)
    try:
        for epoch in tqdm(range(epochs)):
            log_probs = []
            values = []
            rewards = []
            state = env.reset()
            done = False
            steps = 0
            entropy = 0
            h = None
            while not done:
                state = torch.tensor(state, dtype=torch.float, device=device)
                if recurrent_model:
                    (action_probs, h, value) = actor_critic(state, h)
                else:
                    action_probs, value = actor_critic(state)
                policy_dist = torch.distributions.Categorical(probs=action_probs)
                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action)
                entropy += policy_dist.entropy().mean()
                state, reward, done, _ = env.step(action.item())

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)

                if steps == 100000:
                    print('Max number of steps {} reached, breaking episode.'.format(steps))
                    break
                steps += 1
            Qval = 0
            Qvals = np.zeros(len(values))
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + gamma * Qval
                Qvals[t] = Qval

            # update actor critic
            Qvals = torch.FloatTensor(Qvals).to(device)
            values = torch.cat(values)
            log_probs = torch.stack(log_probs)

            if regularize_returns:
                Qvals = F.normalize(Qvals, dim=0)

            advantage = Qvals - values
            actor_loss = (-log_probs * advantage.detach()).sum()
            critic_loss = F.smooth_l1_loss(values, Qvals, reduction='sum')
            loss = actor_loss + critic_coeff * critic_loss - entropy_coeff * entropy

            if logger is not None:
                log_dict = {
                    'Total Loss' : loss.item(),
                    'Actor Loss' : actor_loss.item(),
                    'Critic Loss' : critic_loss.item(),
                    'Entropy' : entropy.item(),
                    'Total reward' : np.sum(rewards),
                    'Episode length' : steps
                }
                #wandb.log(log_dict)
                logger.add_scalar(f'{tag}/Loss/Train', loss.item() / steps, epoch)
                logger.add_scalar(f'{tag}/Actor Loss/Train', actor_loss.item() / steps, epoch)
                logger.add_scalar(f'{tag}/Critic Loss/Train', critic_loss.item() / steps, epoch)
                logger.add_scalar(f'{tag}/Entropy/Train', entropy.item() / steps, epoch)
                logger.add_scalar(f'{tag}/Reward/Train', np.sum(rewards), epoch)
                logger.add_scalar(f'{tag}/Episode Length/Train', steps, epoch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if test_spacing > 0:
                if test_func is not None and epoch % test_spacing == 0 and epoch > 0:
                    test_func(2)

    except KeyboardInterrupt:
        env.close()
        checkpoint = {
            'model': actor_critic,
            'state_dict': actor_critic.state_dict()
        }
        torch.save(checkpoint, f'agents/aborted-agent-{tag}.pt')
        print(f'Saved model at: agents/aborted-agent-{tag}.pt\n')
        exit()

    env.close()
    checkpoint = {
        'model': actor_critic,
        'state_dict': actor_critic.state_dict()
    }
    torch.save(checkpoint, f'agents/trained-agent-{tag}.pt')  # save the model for later use
    print(f'Saved model at: agents/trained-agent-{tag}.pt\n')
