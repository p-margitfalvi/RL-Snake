import numpy as np
import torch

class FNNPolicy(torch.nn.Module):

    def __init__(self, layer_connections, output_distribution=False):
        super().__init__()

        layers = []
        n_layers = len(layer_connections)
        # TODO: Read number of rnn layers from JSON
        self.rnn_layer = torch.nn.LSTM(layer_connections[0], layer_connections[1], 2)
        for idx in range(1, n_layers - 1):
            n_in, n_out = layer_connections[idx], layer_connections[idx + 1]
            layer = torch.nn.Linear(n_in, n_out)
            layers.append(layer)
            if not idx == n_layers - 1:
                layers.append(torch.nn.ReLU())
        if output_distribution:
            layers.append(torch.nn.Softmax(dim=-1))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, h=None):
        x_out, h_out = self.rnn_layer(x, h)
        return self.layers(x_out), h_out

class CNNPolicy(torch.nn.Module):

    def __init__(self, cnn_dict, fnn_layers, output_distribution=False):
        super().__init__()

        cnn_layers = []

        n_cnn_layers = len(cnn_dict['channels'])
        assert cnn_dict['channels'][0] == 2, "First layer has to be 2 channels but got " + str(cnn_dict['channels'][0]) + " channels instead.\n"
        assert (n_cnn_layers == len(cnn_dict['kernel_sizes']) + 1 and n_cnn_layers == len(cnn_dict['strides']) + 1), "CNN_dict error. The channel array should be one longer than the stride/ kernel array."
        for idx in range(n_cnn_layers - 1):
            c_in, c_out = cnn_dict['channels'][idx], cnn_dict['channels'][idx + 1]
            kernel_size, stride = cnn_dict['kernel_sizes'][idx], cnn_dict['strides'][idx]
            layer = torch.nn.Conv2d(c_in, c_out, kernel_size, stride)
            cnn_layers.append(layer)
            cnn_layers.append(torch.nn.ReLU())
        self.cnn_layers = torch.nn.Sequential(*cnn_layers)
        self.rnn = torch.nn.LSTM(fnn_layers[0], fnn_layers[1], 4)
        layers = []
        n_fnn_layers = len(fnn_layers)
        for idx in range(1, n_fnn_layers - 1):
            n_in, n_out = fnn_layers[idx], fnn_layers[idx + 1]
            layer = torch.nn.Linear(n_in, n_out)
            layers.append(layer)
            if not idx == n_fnn_layers - 1:
                layers.append(torch.nn.ReLU())
        if output_distribution:
            layers.append(torch.nn.Softmax(dim=-1))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, h):
        x_ = self.cnn_layers(x)
        x_, h_ = self.rnn(x_.reshape(1, 1, -1), h)
        return self.layers(x_), h_

class Actor_Critic(torch.nn.Module):

    def __init__(self, critic_dict, actor_dict):
        super().__init__()

        # Critic architecture
        cnn_dict = critic_dict['CNN_layers']
        n_cnn_layers = len(cnn_dict['channels'])

        assert cnn_dict['channels'][0] == 2, \
            "First layer has to be 2 channels but got " + str(cnn_dict['channels'][0]) + " channels instead.\n"

        if not n_cnn_layers == 1:
            assert (n_cnn_layers == len(cnn_dict['kernel_sizes']) + 1 and n_cnn_layers == len(cnn_dict['strides']) + 1), \
                "CNN_dict error. The channel array should be one longer than the stride/ kernel array."
            assert critic_dict['FNN_layers'][-1] == 1, \
                "Critic should output only a single Q value"

        cnn_layers = []
        for idx in range(n_cnn_layers - 1):
            c_in, c_out = cnn_dict['channels'][idx], cnn_dict['channels'][idx + 1]
            kernel_size, stride = cnn_dict['kernel_sizes'][idx], cnn_dict['strides'][idx]
            layer = torch.nn.Conv2d(c_in, c_out, kernel_size, stride, groups=2)
            cnn_layers.append(layer)
            cnn_layers.append(torch.nn.ReLU())

        if cnn_layers == []:
            cnn_layers.append(torch.nn.Identity())

        self.critic_cnn = torch.nn.Sequential(*cnn_layers)
        self.critic_rnn = torch.nn.RNN(critic_dict['FNN_layers'][0], critic_dict['FNN_layers'][1], 1)

        n_fnn_layers = len(critic_dict['FNN_layers'])
        fnn_layers = []
        for idx in range(1, n_fnn_layers - 1):
            n_in, n_out = critic_dict['FNN_layers'][idx], critic_dict['FNN_layers'][idx + 1]
            fnn_layers.append(torch.nn.Linear(n_in, n_out))
            if not idx == n_fnn_layers - 2:
                fnn_layers.append(torch.nn.ReLU())
        self.critic_fnn = torch.nn.Sequential(*fnn_layers)

        # Actor architecture
        cnn_dict = actor_dict['CNN_layers']
        n_cnn_layers = len(cnn_dict['channels'])

        assert cnn_dict['channels'][0] == 2,\
            "First layer has to be 2 channels but got " + str(cnn_dict['channels'][0]) + " channels instead.\n"

        if not n_cnn_layers == 1:
            assert (n_cnn_layers == len(cnn_dict['kernel_sizes']) + 1 and n_cnn_layers == len(cnn_dict['strides']) + 1),\
                "CNN_dict error. The channel array should be one longer than the stride/ kernel array."

        cnn_layers = []
        for idx in range(n_cnn_layers - 1):
            c_in, c_out = cnn_dict['channels'][idx], cnn_dict['channels'][idx + 1]
            kernel_size, stride = cnn_dict['kernel_sizes'][idx], cnn_dict['strides'][idx]
            layer = torch.nn.Conv2d(c_in, c_out, kernel_size, stride, groups=2)
            cnn_layers.append(layer)
            cnn_layers.append(torch.nn.ReLU())

        if cnn_layers == []:
            cnn_layers.append(torch.nn.Identity())

        self.actor_cnn = torch.nn.Sequential(*cnn_layers)
        self.actor_rnn = torch.nn.RNN(actor_dict['FNN_layers'][0], actor_dict['FNN_layers'][1], 1)

        n_fnn_layers = len(actor_dict['FNN_layers'])
        fnn_layers = []
        for idx in range(1, n_fnn_layers - 1):
            n_in, n_out = actor_dict['FNN_layers'][idx], actor_dict['FNN_layers'][idx + 1]
            fnn_layers.append(torch.nn.Linear(n_in, n_out))
            if not idx == n_fnn_layers - 2:
                fnn_layers.append(torch.nn.ReLU())
        fnn_layers.append(torch.nn.Softmax(dim=-1))

        self.actor_fnn = torch.nn.Sequential(*fnn_layers)

    def forward(self, x, h_actor, h_critic):

        if len(x.shape) == 3:
            x.unsqueeze_(0)
        """
        # Critic forward pass
        val_ = self.critic_cnn(x)
        val_, h_c_ = self.critic_rnn(val_.reshape(1, 1, -1), h_critic)
        val_ = self.critic_fnn(val_)

        # Actor forward pass
        dist_ = self.actor_cnn(x)
        dist_, h_a_ = self.actor_rnn(dist_.reshape(1, 1, -1), h_actor)
        dist_ = self.actor_fnn(dist_)"""

        x_ = self.actor_cnn(x)
        x_, h_a_ = self.actor_rnn(x_.reshape(1, 1, -1), h_actor)

        dist_ = self.actor_fnn(x_)
        val_ = self.critic_fnn(x_)

        h_c_ = None

        return (dist_.reshape(-1), h_a_), (val_.reshape(1), h_c_)


# Calculates the size of an output after going through a given convolutional layers
def __convolution_output_size__(input_size, kernel_sizes, strides):
    assert len(kernel_sizes) == len(strides)
    output_size = input_size
    for idx, (stride, kernel_size) in enumerate(zip(strides, kernel_sizes)):
        output_size += np.prod(strides[:idx])*(1 - kernel_size)
    return output_size / np.prod(strides)


