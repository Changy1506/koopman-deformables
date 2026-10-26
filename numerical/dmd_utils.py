import os
import numpy as np
import torch

# Root of the separately released dataset (X.pt / Y.pt snapshot matrices).
# Override with the KOOPMAN_DATASET environment variable.
DATASET_ROOT = os.environ.get(
    "KOOPMAN_DATASET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"),
)
PRETRAINED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")


def load_snapshots(name):
    """
    Returns X, Y as numpy float32 arrays of shape (6n, T).
    X[:, t] is the state [u, v] of all n vertices at frame t, Y[:, t] the state at t + 1.
    """
    X_path = os.path.join(DATASET_ROOT, name, "X.pt")
    Y_path = os.path.join(DATASET_ROOT, name, "Y.pt")
    if not os.path.exists(X_path):
        raise FileNotFoundError(
            X_path + " not found. Download the dataset (see README.md) or set KOOPMAN_DATASET."
        )
    X = torch.load(X_path, weights_only=True).numpy()
    Y = torch.load(Y_path, weights_only=True).numpy()
    return X, Y


def pretrained_path(name):
    return os.path.join(PRETRAINED_ROOT, name + ".npz")


def save_pretrained(name, modes, eigs, init_state=None):
    os.makedirs(PRETRAINED_ROOT, exist_ok=True)
    out = {"modes": modes, "eigs": eigs}
    if init_state is not None:
        out["init_state"] = init_state
    np.savez(pretrained_path(name), **out)
    print("Saved to", pretrained_path(name))


def load_pretrained(name, path=None):
    """
    Returns (modes, eigs, init_state). init_state is None when the example
    starts from rest (all-zero state).
    """
    data = np.load(path or pretrained_path(name))
    init_state = data["init_state"] if "init_state" in data else None
    return data["modes"], data["eigs"], init_state
