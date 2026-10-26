"""
Fit the low-rank Koopman operator (exact DMD) of one example from its
snapshot matrices and store modes / eigenvalues in pretrained/<example>.npz.

Usage:
    python fit_dmd.py duck
    python fit_dmd.py all
"""
import sys
import numpy as np
from pydmd import DMD

from dmd_utils import load_snapshots, save_pretrained

# example -> (dataset folder, svd rank, initial state used by the demo)
# initial state: None = start from rest, "first" = X[:, 0], "first*2" = 2 * X[:, 0],
#                "third" = X[:, T // 3]
CONFIGS = {
    "armadillo":    ("armadillo",    50, None),
    "bar":          ("bar",          50, None),
    "duck":         ("duck",         50, "first*2"),
    "duck_novel":   ("duck",         50, "first*2"),  # positions only (no velocity), ablation
    "flamingo":     ("flamingo",     25, None),
    "gummybear":    ("gummybear",    30, "first"),
    "mouse":        ("mouse",        50, "first"),
    "mushroom":     ("mushroom",     50, None),
    "robot_finger": ("robot_finger", 50, None),
    "whale":        ("whale",        50, "first"),
}


def drop_velocity(X):
    X = X.reshape(-1, 6, X.shape[-1])[:, :3, :]
    return X.reshape(-1, X.shape[-1])


def fit(name):
    folder, rank, init = CONFIGS[name]
    X, Y = load_snapshots(folder)
    if name == "duck_novel":
        X, Y = drop_velocity(X), drop_velocity(Y)
    print(name, X.shape, Y.shape)

    dmd = DMD(svd_rank=rank)
    dmd.fit(X, Y)

    if init is None:
        init_state = None
    elif init == "first":
        init_state = X[:, 0]
    elif init == "first*2":
        init_state = X[:, 0] * 2
    elif init == "third":
        init_state = X[:, int(X.shape[1] / 3)]

    save_pretrained(name, dmd.modes, dmd.eigs, init_state)


if __name__ == "__main__":
    if len(sys.argv) < 2 or (sys.argv[1] != "all" and sys.argv[1] not in CONFIGS):
        print("usage: python fit_dmd.py <" + "|".join(CONFIGS) + "|all>")
        sys.exit(1)
    for name in (CONFIGS if sys.argv[1] == "all" else [sys.argv[1]]):
        fit(name)
