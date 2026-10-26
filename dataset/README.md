# Dataset

Training snapshots for *Low-Rank Koopman Deformables with Log-Linear Time Integration*.
The data is not part of this repository; it is available on request (email
changyue.chang@mail.utoronto.ca or open a GitHub issue). Extract it so this folder looks like the layout below, or put it
anywhere and set `KOOPMAN_DATASET=/path/to/dataset`.

You only need it to **train / fit**. All demo scripts run from the pretrained weights in
`numerical/pretrained/` and `neural/pretrained/`.

## Format

Every file is a single `torch.float32` tensor saved with `torch.save`:

```python
X = torch.load("dataset/duck/X.pt", weights_only=True)   # shape (6n, T)
```

* Column `t` of `X` is the full state at frame `t`, stored per vertex as
  `[u_x, u_y, u_z, v_x, v_y, v_z]`. `u` is the displacement from the rest shape and
  `v` the velocity (the difference between consecutive displacements), so `X.reshape(n, 6, T)`.
* `Y[:, t]` is the state one time step after `X[:, t]`. The pairs `(X, Y)` are what DMD
  and the neural model are fit to.
* Vertex order matches the rest shape in the code repository (`x` in the matching `.h5`
  file under `numerical/data/` or `neural/data/`).
* The `*_video` examples are 2D point tracks from real videos, stored per point as
  `[u_x, u_y, v_x, v_y]` (shape `(4m, T)`). The scripts pad them to 3D.

## Layout

| folder | files | used by | description |
|---|---|---|---|
| `armadillo/` | `X.pt`, `Y.pt` | numerical | armadillo under a body load |
| `bar/` | `X.pt`, `Y.pt` | numerical | twisted bar |
| `duck/` | `X.pt`, `Y.pt` | numerical + neural | rubber duck, poked |
| `flamingo/` | `X.pt`, `Y.pt` | numerical | flamingo, pushed |
| `gummybear/` | `X.pt`, `Y.pt` | numerical + neural | free vibration of a gummy bear |
| `mouse/` | `X.pt`, `Y.pt` | numerical | mouse |
| `mushroom/` | `X.pt`, `Y.pt` | numerical | mushroom under a body load |
| `robot_finger/` | `X.pt`, `Y.pt` | numerical | 3-chamber pneumatic finger |
| `whale/` | `X.pt`, `Y.pt` | numerical | whale |
| `finger/` | `X_<geo>.pt`, `Y_<geo>.pt` (58 shapes) | neural | inflatable finger family, `<geo>` = `gap_wall_a0_amid_a1` |
| `frog/` | `X_<geo>.pt`, `Y_<geo>.pt` (24 shapes) | neural | inflatable frog family, `<geo>` = `length_bend_taper_zoffset` |
| `duck_video/` | `X_{1,2,3}.pt`, `Y_{1,2,3}.pt` | neural | real-world rubber duck, optical-flow tracks |
| `damp_video/` | `X_{1,2}.pt`, `Y_{1,2}.pt` | neural | real-world damped blob, optical-flow tracks |

For `finger/` and `frog/`, the rest shape of each geometry is `neural/data/<name>/h5/<geo>.h5`
in the code repository.
