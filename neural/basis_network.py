import torch
import torch.nn as nn
import math
class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # first layer init
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features,
                     1 / self.linear.in_features
                )
            else:
                # SIREN init
                bound = math.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class basis_network(nn.Module):
    def __init__(self, g, r, hidden=128, depth=5):
        super().__init__()

        in_dim = 3 + g

        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.ELU()]
            #layers += [SirenLayer(dim, hidden, omega_0=30.0, is_first=(dim == in_dim))]
            dim = hidden
        self.backbone = nn.Sequential(*layers)

        self.out_real = nn.Linear(hidden, r * 6)
        self.out_imag = nn.Linear(hidden, r * 6)

        self.r = r

    def forward(self, x):
        """
        x: (B, 3+g)
        return: (B, r, 3) complex
        """
        h = self.backbone(x)
        real = self.out_real(h)#.view(-1, 6, self.r)
        imag = self.out_imag(h)#.view(-1, 6, self.r)
        return torch.complex(real, imag)
