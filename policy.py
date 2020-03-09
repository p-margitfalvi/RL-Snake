import numpy as np
import torch

class FNNPolicy(torch.nn.Module):

    def __init__(self, layer_connections, output_distribution=False):
        super().__init__()

        layers = []
        n_layers = len(layer_connections)
        # TODO: Read number of rnn layers from JSON
        self.rnn_layer = torch.nn.RNN(layer_connections[0], layer_connections[1], num_layers=3, nonlinearity='relu')
        for idx in range(1, n_layers - 1):
            n_in, n_out = layer_connections[idx], layer_connections[idx + 1]
            layer = torch.nn.Linear(n_in, n_out)
            layers.append(layer)
            if not idx == n_layers - 1:
                layers.append(torch.nn.ReLU())
        if output_distribution:
            layers.append(torch.nn.Softmax(dim=0))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, h=None):
        x_out, h_out = self.rnn_layer(x, h)
        return self.layers(x_out), h_out

class CNNPolicy(torch.nn.Module):
    def __init__(self, cnn_dict, fnn_layers, output_distribution=False):
        super().__init__()

        layers = []

        n_cnn_layers = len(cnn_dict['channels'])
        assert cnn_dict['channels'][0] == 2, "First layer has to be 2 channels but got " + str(cnn_dict['channels'][0]) + " channels instead.\n"
        for idx in range(n_cnn_layers - 1):
            c_in, c_out = cnn_dict['channels'][idx], cnn_dict['channels'][idx + 1]
            kernel_size, stride = cnn_dict['kernel_sizes'][idx], cnn_dict['strides'][idx]
            layer = torch.nn.Conv2d(c_in, c_out, kernel_size, stride)
            layers.append(layer)
            if not idx == n_cnn_layers - 1:
                layers.append(torch.nn.ReLU())
        n_fnn_layers = len(fnn_layers)


        for idx in range(n_fnn_layers - 1):
            n_in, n_out = fnn_layers[idx], fnn_layers[idx + 1]
            layer = torch.nn.Linear(n_in, n_out)
            layers.append(layer)
            if not idx == n_fnn_layers - 1:
                layers.append(torch.nn.ReLU())
        if output_distribution:
            layers.append(torch.nn.Softmax(dim=0))

        self.layers = torch.nn.Sequential(*layers)

        # TODO: Implement CNN model

# Calculates the size of an output after going through a given convolutional layers
def __convolution_output_size__(input_size, kernel_sizes, strides):
    assert len(kernel_sizes) == len(strides)
    output_size = input_size
    for idx, (stride, kernel_size) in enumerate(zip(strides, kernel_sizes)):
        output_size += np.prod(strides[:idx])*(1 - kernel_size)
    return output_size / np.prod(strides)


