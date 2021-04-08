import torch
import torch.nn.functional as F

class SimpleA2C(torch.nn.Module):

    def __init__(self, input_size, output_size):

        super().__init__()

        self.actor_layers = torch.nn.Sequential(
                torch.nn.Linear(input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_size)
                )

        self.critic_layers = torch.nn.Sequential(
                torch.nn.Linear(input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
                )

    def forward(self, x):

        x_actor = self.actor_layers(x)
        x_actor = F.softmax(x_actor, dim=0) # TODO: change dim

        x_critic = self.critic_layers(x)

        return x_actor, x_critic
