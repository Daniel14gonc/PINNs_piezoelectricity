"""Network architecture for the direct PINN."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.init as init
from torch import nn

from ..config import HEIGHT


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def init_weights_siren(m, omega_0: float = 30.0):
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        with torch.no_grad():
            m.weight.uniform_(-np.sqrt(6 / num_input) / omega_0,
                              np.sqrt(6 / num_input) / omega_0)


class SinActivation(nn.Module):
    """Wrapper to use ``torch.sin`` inside an ``nn.Sequential``."""

    def forward(self, x):
        return torch.sin(x)


class FCN(nn.Module):
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

        x_norm = x[:, 0:1] / 1
        u_modified = x_norm * u
        v_modified = x_norm * v
        phi_modified = (x[:, 1:2] / HEIGHT) * phi

        return torch.cat([u_modified, v_modified, phi_modified, outputs[:, 3:]],
                         dim=1)


def build_default_model(device=None,
                        input_size: int = 2,
                        output_size: int = 8,
                        hidden_sizes=(100, 250)):
    """Reproduce the model build sequence used in PINN_pz_v3_directo.ipynb."""
    model = FCN(input_size, list(hidden_sizes), output_size)
    model.apply(init_weights)
    if device is not None:
        model.to(device)
    return model
