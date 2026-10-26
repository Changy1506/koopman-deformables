import sys
import numpy as np
import h5py
import polyscope as ps
from scipy.linalg import pinv
import polyscope.imgui as psim
from dmd_utils import load_pretrained

with h5py.File("data/gummybear/gummybear.h5", "r") as f:
    x = f["x"][:]
    f = f["faces"][:]
print(x.shape, f.shape)


# modes / eigenvalues from fit_dmd.py (optionally pass another .npz path)
modes, eigs, init_state = load_pretrained("gummybear", sys.argv[1] if len(sys.argv) > 1 else None)

state = init_state

mode_inv = pinv(modes)

def predict(X):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs ** 1), mode_inv, X]
        )

def callback():
    global state, dmd
    state_now = state.reshape(-1,6)
    ps.register_surface_mesh("gummybear", state_now[:,:3] + x, f - 1)


    state = predict(state)


ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)
ps.show()




