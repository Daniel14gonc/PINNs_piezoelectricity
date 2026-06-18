"""Network architectures for the indirect PINN."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.init as init
from torch import nn

from .. import config
from ..config import HEIGHT, VOLTAGE


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def u_constraint(x, y):
    return 0  # x * nn.functional.relu(x)  # u = 0 at x = 0


def v_constraint(x, y):
    return 0  # x * nn.functional.relu(x)  # v = 0 at x = 0


def phi_constraint(x, y):
    # Reads ``config.VOLTAGE`` at call time so the applied voltage can be
    # changed (e.g. the generalization study, Cluster 8) by setting
    # ``pinn_piezo.config.VOLTAGE`` before building/evaluating the model.
    return config.VOLTAGE / HEIGHT * y  # φ = 0 at y = 0, φ = V at y = H


class FCNUniform(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        activation = nn.Tanh

        layers = [
            ('input', nn.Linear(input_size, hidden_size)),
            ('act0', activation()),
        ]
        for i in range(num_layers):
            layers.append((f'hidden_{i}', nn.Linear(hidden_size, hidden_size)))
            layers.append((f'act_{i}', activation()))
        layers.append(('output', nn.Linear(hidden_size, output_size)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        outputs = self.net(x)
        u, v, phi = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

        u_modified = x[:, 0:1] * u
        v_modified = x[:, 0:1] * v
        phi_modified = (x[:, 1:2] * (x[:, 1:2] - HEIGHT) * phi
                        + phi_constraint(x[:, 0:1], x[:, 1:2]))

        return torch.cat([u_modified, v_modified, phi_modified, outputs[:, 3:]],
                         dim=1)


class FCNPyramid(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.Tanh):
        super().__init__()

        layers = [
            ('input', nn.Linear(input_size, hidden_sizes[0])),
            ('act0', activation()),
        ]
        for i in range(1, len(hidden_sizes)):
            layers.append((f'hidden_{i - 1}',
                           nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])))
            layers.append((f'act_{i}', activation()))
        layers.append(('output', nn.Linear(hidden_sizes[-1], output_size)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        outputs = self.net(x)
        u, v, phi = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

        u_modified = x[:, 0:1] * u
        v_modified = x[:, 0:1] * v
        phi_modified = (x[:, 1:2] * (x[:, 1:2] - HEIGHT) * phi
                        + phi_constraint(x[:, 0:1], x[:, 1:2]))

        return torch.cat([u_modified, v_modified, phi_modified, outputs[:, 3:]],
                         dim=1)


def get_model(input_size, hidden_sizes, output_size, type='uniform'):
    if type == 'uniform':
        return FCNUniform(input_size, hidden_sizes, output_size)
    return FCNPyramid(input_size, hidden_sizes, output_size)


def build_default_model(device=None,
                        model_type: str = 'pyramid',
                        input_size: int = 2,
                        output_size: int = 8,
                        hidden_sizes=(100, 250)):
    """Reproduce the model build sequence used in PINN_pz_v3.ipynb."""
    if model_type == 'pyramid':
        model = get_model(input_size, list(hidden_sizes), output_size,
                          type=model_type)
    else:
        # original notebook called: get_model((2, 300, 3, 8), type='uniform')
        model = get_model((2, 300, 3, 8), type=model_type)
    model.apply(init_weights)
    if device is not None:
        model.to(device)
    return model
