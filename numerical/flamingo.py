import sys
import numpy as np
import h5py
import polyscope as ps
from scipy.linalg import pinv
import polyscope.imgui as psim
from dmd_utils import load_pretrained
with h5py.File("data/flamingo/flamingo.h5", "r") as f:
    x = f["x"][:]
    f = f["faces"][:]
print(x.shape, f.shape)


# modes / eigenvalues from fit_dmd.py (optionally pass another .npz path)
modes, eigs, _ = load_pretrained("flamingo", sys.argv[1] if len(sys.argv) > 1 else None)

state = np.zeros(modes.shape[0], dtype=np.float32)

mode_inv = pinv(modes)

def predict(X):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs), mode_inv, X]
        )

vel_x = 0
vel_y = 0

def callback():
    global state, dmd
    global vel_x, vel_y
    state_now = state.reshape(-1,6).real
    ps.register_surface_mesh("flamingo", state_now[:,:3] + x, f - 1)

    _, vel_x = psim.SliderFloat("vel_x", vel_x, v_min=-1, v_max=1)
    _, vel_y = psim.SliderFloat("vel_y", vel_y, v_min=-1, v_max=1)

    if(psim.Button("Poke")):
        
        state = state.reshape(-1,6)
        state[:,3] += vel_x / 50. * 2
        state[:,5] += vel_y / 50. * 2
        state = state.reshape(-1)


    state = predict(state)



ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)
ps.show()




