import sys
import numpy as np
import h5py
import polyscope as ps
from scipy.linalg import pinv
from dmd_utils import load_pretrained

with h5py.File("data/whale/whale.h5", "r") as f:
    x = f["x"][:].T
    f = f["faces"][:].T
print(x.shape, f.shape)


# modes / eigenvalues from fit_dmd.py (optionally pass another .npz path)
modes, eigs, init_state = load_pretrained("whale", sys.argv[1] if len(sys.argv) > 1 else None)

state = init_state

mode_inv = pinv(modes)

def predict(X, step_size = 500):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs ** step_size), mode_inv, X]
        )

step_size = 1
idx = 0

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0,  0.0],
        [0.0,   c, -s  ],
        [0.0,   s,  c  ]
    ], dtype=float)

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c]
    ], dtype=float)


omega_z = 4.0   # radians per unit-t around Z
omega_y = 0.   # radians per unit-t around Y
vx      = 0.35   # units per unit-t along +X

pivot = x.mean(axis=0) + np.array([0,0,0.075])

def callback():
    global state, dmd
    global step_size
    global idx, pivot
    state_now = state.reshape(-1,6)
    t = idx / 100.0

    R = rot_x(omega_z * t) @ rot_y(omega_y * t)  

    mesh = ps.register_surface_mesh("whale", (state_now[:,:3] + x - pivot)  @ R.T + pivot + np.array([0.0, vx * t * 10 - 0.5 * t * t * 10, vx * t * 3])
, f - 1)

    idx = idx + 1

    state = predict(state, step_size)

    



ps.init()
ps.look_at((-4,1,-4), (-2,0,0))
ps.set_ground_plane_mode("shadow_only")
#ps.set_environment_map("studio") 
ps.set_user_callback(callback)
ps.show()




