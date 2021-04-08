import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import os

#wandb.init(project= 'rl-snake')
__DEFAULT_TENSOR__ = torch.double


class Agent():

    def __init__(self, env, discount=0.99):
        self.env = env
        self.discount = discount

    def rollout(self, rollout_len=None, explore=True):
        pass

class A2CAgent(Agent):

    @classmethod
    def load(cls, path, env, device='cpu', logger=None):
        checkpoint = torch.load(path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])

        return cls(env, model, checkpoint['optim'],
        checkpoint['entropy_coeff'], checkpoint['critic_coeff'],
        checkpoint['batch_size_ep'], device,
        checkpoint['discount'], logger)

    def __init__(self, env, model, optim, entropy_coeff, critic_coeff,
            batch_size_ep, n_optim=1,  device='cpu', discount=0.99,
            logger=None):

        super().__init__(env, discount)

        self.env = env
        self.model = model
        self.optim = optim

        self.batch_size = batch_size_ep
        self.n_optim = n_optim

        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.discount = discount

        self.device = device
        self.logger = logger

    def rollout(self, rollout_len=None, explore=True, render=False):

        if rollout_len is None:
            rollout_len = np.inf

        state = self.env.reset()
        done = False
        rollout_buffer = []

        step = 0

        while not done and step < rollout_len:

            with torch.no_grad():
                action_probs, value = self.model(torch.as_tensor(state.flatten()))
                action_dist = torch.distributions.Categorical(probs=action_probs)
                action = action_dist.sample() if explore else action_probs.argmax()

            next_state, reward, done, info = self.env.step(int(action))
            if render:
                self.env.render()
            transition = (state, action, reward, next_state, done)
            state = next_state

            rollout_buffer.append(transition)
            step += 1

        return rollout_buffer

    def test(self, n):
        for i in range(n):
            self.rollout(render=True, explore=False)

    def fill_batch(self):
        batch = []

        for ep in range(self.batch_size):
            batch.append(self.rollout())

        return batch

    def save_model(self, tag):
        checkpoint = {  'model': self.model,
                        'state_dict': self.model.state_dict(),
                        'optim': self.optim,
                        'critic_coeff': self.critic_coeff,
                        'entropy_coeff':self.entropy_coeff,
                        'batch_size_ep': self.batch_size,
                        'discount': self.discount}
        torch.save(checkpoint, f'agents/agents-checkpoint-{tag}.pt')

    def train(self, steps, save_freq=0, keep_num_chckpt=None):
        self.model.train()

        saved_model_tags = []

        for n_steps in tqdm(range(steps)):

            batch = self.fill_batch()

            for optim in range(self.n_optim):
                value_loss = torch.zeros(1, requires_grad=True)
                policy_loss = torch.zeros(1, requires_grad=True)
                entropy_loss = torch.zeros(1, requires_grad=True)

                n_samples = 0

                for trajectory in batch:
                    v_terminal = 0
                    if not trajectory[-1][4]: # If last step is not done
                        _, v_terminal = self.model(torch.as_tensor(trajectory[-1][0].flatten())) # Bootstrap

                    traj_return = v_terminal
                    n_samples += len(trajectory)

                    rw_sum = 0
                    for transition in reversed(trajectory):
                        rw_sum += transition[2]

                        traj_return = transition[2] + self.discount * traj_return
                        action_probs, value = self.model(torch.as_tensor(transition[0].flatten()))
                        entropy = -torch.sum(action_probs*torch.log(action_probs))

                        value_loss = value_loss + (traj_return - value)**2
                        policy_loss = policy_loss - torch.log(action_probs[transition[1]])*(traj_return - value.detach())
                        entropy_loss = entropy_loss - entropy

                policy_term = policy_loss / n_samples
                value_term = self.critic_coeff * value_loss / n_samples
                entropy_term = self.entropy_coeff * entropy_loss / n_samples

                loss = policy_term + value_term + entropy_term
                loss.backward()

                param_norms = []
                for p in list(filter(lambda p: p.grad is not None, self.model.parameters())):
                    param_norms.append(p.grad.data.norm(2).item())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.6)
                self.optim.step()
                self.optim.zero_grad()

                if self.logger is not None:
                    log_dict = {
                        'Total Loss' : loss.item(),
                        'Actor Loss' : policy_term.item(),
                        'Critic Loss' : value_term.item(),
                        'Total reward' : rw_sum,
                        'Entropy' : entropy_term,
                        'Episode Length': len(trajectory),
                        'Grad Norm': sum(param_norms) / len(param_norms)
                }

                self.logger.log(log_dict)

            if n_steps % save_freq == 0:
                tag = f'step-{n_steps}'
                self.save_model(tag)

                if keep_num_chckpt == len(saved_model_tags):
                    # Delete oldest model
                    os.remove(f'agents/agents-checkpoint-{saved_model_tags.pop(0)}.pt')

                saved_model_tags.append(tag)


        self.save_model('final')
