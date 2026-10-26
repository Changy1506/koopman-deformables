import sys
import numpy as np
import h5py
import polyscope as ps
from scipy.linalg import pinv
from dmd_utils import load_pretrained

with h5py.File("data/mouse/mouse.h5", "r") as f:
    x = f["x"][:]
    f = f["faces"][:]
print(x.shape, f.shape)


# modes / eigenvalues from fit_dmd.py (optionally pass another .npz path)
modes, eigs, init_state = load_pretrained("mouse", sys.argv[1] if len(sys.argv) > 1 else None)

state = init_state

mode_inv = pinv(modes)
eigs = eigs * 0.98

def predict(X, step_size = 1):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs ** step_size), mode_inv, X]
        )

step_size = 5


def callback():
    global state, dmd
    global step_size
    state_now = state.reshape(-1,6)
    mesh = ps.register_surface_mesh("mouse", state_now[:,:3] + x, f - 1)

    state = predict(state, step_size)



ps.init()
ps.set_ground_plane_mode("shadow_only")
#ps.set_environment_map("studio") 
ps.set_user_callback(callback)
ps.show()




