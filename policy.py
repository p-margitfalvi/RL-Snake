import torch

class FNNPolicy(torch.nn.Module):

    def __init__(self, layer_connections, output_distribution=False):
        super().__init__()

        layers = []
        n_layers = len(layer_connections)
        for idx in range(n_layers - 1):
            n_in, n_out = layer_connections[idx], layer_connections[idx + 1]
            layer = torch.nn.Linear(n_in, n_out)
            layers.append(layer)
            if not idx == n_layers - 1:
                layers.append(torch.nn.ReLU())
        if output_distribution:
            layers.append(torch.nn.Softmax(dim=0))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CNNPolicy(torch.nn.Module):
    def __init__(self, cnn_dict, fnn_layers):
        super().__init__()

        # TODO: Implement CNN model
