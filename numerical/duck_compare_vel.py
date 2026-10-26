import numpy as np
import h5py
import polyscope as ps
from scipy.linalg import pinv
import polyscope.imgui as psim
from dmd_utils import load_pretrained

with h5py.File("data/duck/duck.h5", "r") as f:
    x = f["x"][:]
    f = f["faces"][:]
print(x.shape, f.shape)


# Ablation: DMD on positions only (duck_novel, see fit_dmd.py) vs. positions + velocities (duck)
noVel = False

modes, eigs, init_state = load_pretrained("duck_novel" if noVel else "duck")

state = init_state

mode_inv = pinv(modes)

def predict(X):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs), mode_inv, X]
        )

def callback_with_vel():
    global state, dmd
    state_now = state.reshape(-1,6).real
    ps.register_surface_mesh("duck", state_now[:,:3] + x, f - 1)

    state = predict(state)


def callback_no_vel():
    global state, dmd
    state_now = state.reshape(-1,3).real
    ps.register_surface_mesh("duck", state_now + x, f - 1)

    state = predict(state)

ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback_with_vel)
if noVel:
    ps.set_user_callback(callback_no_vel)
ps.show()




