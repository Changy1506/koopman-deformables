import torch
import torch.nn as nn

class eigenvalue_network(nn.Module):
    def __init__(self, g, r, hidden=64, depth=3):
        super().__init__()

        layers = []
        dim = g
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.ELU()]
            dim = hidden
        self.backbone = nn.Sequential(*layers)

        self.out_real = nn.Linear(hidden, r)
        self.out_imag = nn.Linear(hidden, r)

    def forward(self, x):
        h = self.backbone(x)
        real = self.out_real(h)
        imag = self.out_imag(h)
        return torch.complex(real, imag)
