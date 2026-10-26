import os
import torch

# Root of the separately released dataset (X / Y snapshot matrices as .pt).
# Override with the KOOPMAN_DATASET environment variable.
DATASET_ROOT = os.environ.get(
    "KOOPMAN_DATASET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"),
)
PRETRAINED_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
CHECKPOINT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def dataset_path(name, filename):
    return os.path.join(DATASET_ROOT, name, filename)


def has_dataset(name, filename="X.pt"):
    return os.path.exists(dataset_path(name, filename))


def load_snapshots(name, filename):
    """
    Loads one snapshot matrix, e.g. load_snapshots("duck", "X.pt").
    Returns a numpy float32 array of shape (6n, T) (column t = state at frame t).
    """
    path = dataset_path(name, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            path + " not found. Download the dataset (see README.md) or set KOOPMAN_DATASET."
        )
    return torch.load(path, weights_only=True).numpy()


def checkpoint_path(name):
    """Where training_<name>.py writes its weights."""
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    return os.path.join(CHECKPOINT_ROOT, name + "_dmd_models.pt")


def load_checkpoint(name, path=None, map_location="cpu"):
    """
    Loads pretrained/<name>_dmd_models.pt (or `path`, e.g. a checkpoint you trained).
    Keys: eigenvalue_model, basis_model, g, r and, for some examples,
    init_state / init_states / num_frames needed to run the demo without the dataset.
    """
    path = path or os.path.join(PRETRAINED_ROOT, name + "_dmd_models.pt")
    print("Loading", path)
    return torch.load(path, map_location=map_location, weights_only=True)
