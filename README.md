# Low-Rank Koopman Deformables with Log-Linear Time Integration

Official code for

> **Low-Rank Koopman Deformables with Log-Linear Time Integration**
> Yue Chang, Peter Yichen Chen, Eitan Grinspun, Maurizio M. Chiaramonte
> SIGGRAPH 2026 · [arXiv:2602.07687](https://arxiv.org/abs/2602.07687)

We model deformable dynamics with a **low-rank Koopman operator** fit by Dynamic Mode
Decomposition (DMD). The state of a mesh with `n` vertices is the stacked displacement and
velocity of every vertex, `x ∈ R^{6n}`. From snapshot pairs `(X, Y)` with `Y = A X`, DMD
gives `r` complex modes `Φ ∈ C^{6n×r}` and eigenvalues `Λ = diag(λ_1..λ_r)`, and

```
x_{t+k} = Φ Λ^k Φ^+ x_t
```

Jumping `k` steps ahead is a diagonal power, so a whole rollout costs time logarithmic in
`k` instead of linear. Each step costs `O(r)` work in the reduced space, independent of
mesh size. External forces enter as velocity impulses before the propagation.

This repository contains **two implementations** of the method:

| | **Numerical** (`numerical/`) | **Neural, discretization-agnostic** (`neural/`) |
|---|---|---|
| Koopman basis | DMD modes of one mesh (PyDMD, truncated SVD) | a *basis network* `(rest position p, shape code g) → 6×r complex basis`, queried at every vertex |
| Eigenvalues | DMD eigenvalues | an *eigenvalue network* `g → r complex eigenvalues`, with `|λ| ≤ 1` |
| Generalizes to | new forces, directions, step sizes, initial states | new **meshes / resolutions** of the same shape and new **shapes** in a parametric family |
| Fitting | closed form, seconds | gradient descent (reconstruction + one-step losses) |
| Backend | NumPy / SciPy | PyTorch |

Both implementations come with **pretrained weights for every example**, so all demos run
without the training data.

---

## Installation

```bash
git clone https://github.com/Changy1506/koopman-deformables.git
cd koopman-deformables
pip install -r requirements.txt
```

The demos open an interactive [Polyscope](https://polyscope.run) window. Tested on Python 3.8
with PyTorch 2.4 (CPU). A GPU is used automatically by the neural scripts if available.

## Dataset (only needed for training)

The training snapshots (about 12 GB) are **available on request**: email Yue Chang at
[changyue.chang@mail.utoronto.ca](mailto:changyue.chang@mail.utoronto.ca) or open a GitHub
issue. You don't need them to run the demos, only to re-train or re-fit.

Once you have the data, extract it into `dataset/` at the repository root, or anywhere else
and point the code to it:

```bash
export KOOPMAN_DATASET=/path/to/dataset     # Windows: set KOOPMAN_DATASET=C:\path\to\dataset
```

Each example is a pair of `torch.float32` tensors `X.pt`, `Y.pt` of shape `(6n, T)`, with
`Y[:, t]` one step after `X[:, t]`. See [dataset/README.md](dataset/README.md) for the full
format and layout. Meshes, rest shapes and other geometry are small and ship with the code in
`numerical/data/` and `neural/data/`.

---

## Repository layout

```
koopman-deformables/
├── numerical/                 # DMD on a fixed mesh
│   ├── fit_dmd.py             # "training": fit DMD from dataset/, write pretrained/<example>.npz
│   ├── dmd_utils.py           # dataset / pretrained I/O
│   ├── <example>.py           # demos (load pretrained/<example>.npz)
│   ├── interactive.py         # trajectory-drawing UI used by robot_finger_interactive.py
│   ├── data/<example>/        # rest meshes (.h5 / .mesh), chamber face indices, ...
│   └── pretrained/            # fitted modes + eigenvalues for every example
├── neural/                    # discretization-agnostic neural extension
│   ├── basis_network.py       # (p, g) -> complex basis
│   ├── eigenvalue_network.py  # g -> complex eigenvalues
│   ├── dataloader.py
│   ├── data_utils.py          # dataset / checkpoint I/O
│   ├── training_<example>.py  # train from dataset/, write checkpoints/<example>_dmd_models.pt
│   ├── test_<example>.py      # demos (load pretrained/<example>_dmd_models.pt)
│   ├── data/<example>/        # rest meshes, per-shape rest geometry (.h5), chamber faces
│   └── pretrained/            # trained weights for every example
└── dataset/                   # <- X.pt / Y.pt snapshots go here (available on request)
```

All scripts use paths relative to their own folder, so run them from `numerical/` or `neural/`.

---

## Numerical implementation

```bash
cd numerical
python duck.py                       # run a demo with the pretrained modes
python fit_dmd.py duck               # re-fit from the dataset (or: python fit_dmd.py all)
python duck.py pretrained/duck.npz   # optional: run with a specific .npz
```

`fit_dmd.py` stores `modes`, `eigs` and, when the demo starts from a recorded frame,
`init_state` in `pretrained/<example>.npz`. The rank of each example is listed in
`fit_dmd.py`. Fitting takes from under a second to about 15 s per example on a CPU.

| paper | demo | what it shows |
|---|---|---|
| Fig. 1 | `whale.py` | teaser: whale deformation from a recorded initial state, drawn on top of a rigid tumbling trajectory |
| Fig. 2 | `gummybear.py` | free vibration of the gummy bear with our model (the implicit-Euler comparison is not included) |
| Fig. 4 | `duck_compare_vel.py` | ablation: DMD on displacements only (`duck_novel`) instead of displacement + velocity. Set `noVel = False` for the stable version |
| Fig. 5 | `mushroom.py` | generalization to a **different load magnitude** than in training (load factor in `apply_force`) |
| Fig. 10 | `armadillo.py` | generalization to a different load magnitude on a strongly nonlinear motion (load factor in `apply_force`) |
| Fig. 11 | `flamingo.py` | generalization to **unseen push directions**: pick a direction with the sliders, then press *Poke* |
| Fig. 13 | `duck.py` | **interactive** poking (velocity sliders + *Poke*) with an adjustable **step size**: `k` steps are taken at once via `Λ^k` |
| Fig. 14 | `mouse.py` | adding **damping without retraining** by scaling the eigenvalues (`0.98 λ`), from a recorded initial state, 5 steps per frame |
| Fig. 15, 16 | `robot_finger_interactive.py` | **control** of the 3-chamber pneumatic finger: draw a goal trajectory on the canvas, the chamber pressures are solved for (NNLS) and the finger follows it |
| Sec. 8 inset (p. 13) | `bar.py` | **limitation**: a bar twisted by prescribed rotations of its two ends (ramped up to 90°, set in `callback`). Under rotational loading larger than in training (2× in the paper), the deformation drifts from the ground truth |

### Fitting a DMD model on your own data

1. Simulate your object with any solver and record, for every frame, the per-vertex
   displacement `u` and velocity `v` (the difference of consecutive displacements). Stack
   them as columns of `X` (frames `0..T-1`) and `Y` (frames `1..T`), each of shape `(6n, T)`
   with rows ordered `[u_x, u_y, u_z, v_x, v_y, v_z]` per vertex, and save them with
   `torch.save(torch.from_numpy(X.astype(np.float32)), "dataset/<name>/X.pt")` (same for `Y`).
2. Add a line to `CONFIGS` in `fit_dmd.py`, e.g. `"<name>": ("<name>", 50, None)`
   (dataset folder, SVD rank, initial state of the demo), then run `python fit_dmd.py <name>`.
3. Load the result in your own script, using any demo above as a template:

```python
from scipy.linalg import pinv
from dmd_utils import load_pretrained

modes, eigs, _ = load_pretrained("<name>")
mode_inv = pinv(modes)
z = mode_inv @ x                    # project a full state x (6n,) to the reduced space
x_next = (modes @ (eigs ** k * z)).real   # jump k steps ahead
```

Forces are applied by adding to the velocity part of `x` before projecting. The rank is the
main knob: higher ranks capture more motion, but past a point the fit gets ill-conditioned.

## Neural implementation (discretization-agnostic extension)

```bash
cd neural
python test_duck_discretization_change.py                                  # demo with pretrained weights
python training_duck.py                                                    # train from the dataset
python test_duck_discretization_change.py checkpoints/duck_dmd_models.pt   # demo with your own weights
```

For a shape code `g` and rest positions `p_i`, the basis network outputs a complex `6×r` block
per vertex, and stacking these gives the basis `B(g)`. The eigenvalue network outputs `λ(g)`,
clamped to `|λ| ≤ 1`. One step is the least-squares projection onto `B`, multiplication by
`diag(λ)`, and reconstruction:

```
ẑ_t = (BᵀB)^{-1} Bᵀ x_t,   x_{t+1} = Re( B diag(λ) ẑ_t )
```

Training minimizes the reconstruction error of `x_t`, the error of the predicted reduced state
against the projection of `x_{t+1}`, and the reconstruction error of `x_{t+1}`. Because the
basis is a field over rest positions, a trained model can be evaluated on **any discretization**
of the shape, and on **new shapes** of a parametric family.

| paper | demo | training script | shape code `g` | what it shows |
|---|---|---|---|---|
| Fig. 3 | `test_duck_discretization_change.py` | `training_duck.py` | none | **discretization-agnostic**: the duck model evaluated on a different tet mesh than it was trained on (`data/duck/duck_{2.5k,5k,10k,50k}.mesh`, set in the script); poke it with the sliders + *Poke* |
| Fig. 7 | `test_gummybear.py` | `training_gummybear.py` | none | the Fig. 7 gummy bear model, interactive poking (sliders + *Poke*) |
| Fig. 8 | `test_frog.py` | `training_frog.py` | 4: chamber length, bend, taper, z-offset | one model for a **family of inflatable frogs** (press *Poke* to toggle the pressure) |
| Fig. 12 | `test_finger.py` | `training_finger.py` | 5: `gap, wall, a0, amid, a1` (file names in `data/finger/h5/`) | one model for a **family of inflatable soft fingers**: press *Poke* to inflate, then it releases |
| Fig. 18 | `test_duck_video_adjust.py` | `training_duck_video.py` | none | **real-world video** of a rubber duck: the model is learned from sparse optical-flow tracks and drives a dense surface mesh |
| Fig. 18 | `test_damp_video_adjust.py` | `training_damp_video.py` | none | real-world video of a damped blob, same as above |

Figures without a script: Fig. 6 (pipeline diagram), Fig. 9 (basis-size study) and Fig. 17
(frog shape optimization). Fig. 7 compares ways of handling the imaginary part; only the
resulting model is included. Fig. 3 in
the paper also shows a 500k-vertex mesh, which is left out of the repository for size.

Notes

* Neural scripts scale displacements by 100 internally (see `dataloader.py`).
* Interactive demos (sliders / *Poke* in the top-right panel) start from rest. The few
  scripted demos that start from a recorded frame (gummy bear, mouse, whale, video examples) read it from the
  pretrained file. When the dataset is present, the video demos also show the ground-truth
  tracks (`GT`) next to the prediction.
* Training writes to `neural/checkpoints/` and never overwrites `neural/pretrained/`.

---

## Citation

```bibtex
@misc{chang2026lowrankkoopman,
  title         = {Low-Rank Koopman Deformables with Log-Linear Time Integration},
  author        = {Chang, Yue and Chen, Peter Yichen and Grinspun, Eitan and Chiaramonte, Maurizio M.},
  year          = {2026},
  eprint        = {2602.07687},
  archivePrefix = {arXiv},
  primaryClass  = {cs.GR},
  note          = {SIGGRAPH 2026}
}
```

## License

The code is released under the [MIT License](LICENSE).
