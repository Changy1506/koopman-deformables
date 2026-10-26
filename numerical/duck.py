import sys
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


# modes / eigenvalues from fit_dmd.py (optionally pass another .npz path)
modes, eigs, _ = load_pretrained("duck", sys.argv[1] if len(sys.argv) > 1 else None)

state = np.zeros(modes.shape[0], dtype=np.float32)

mode_inv = pinv(modes)

def predict(X, step_size = 1):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs ** step_size), mode_inv, X]
        )

vel_x = 0
vel_y = 0
step_size = 1
def callback():
    global state, dmd
    global vel_x, vel_y, step_size
    state_now = state.reshape(-1,6)
    mesh = ps.register_surface_mesh("Duck", state_now[:,:3] + x, f - 1)
    mesh.set_material("duck")
    mesh.set_color((0.85, 0.65, 0.25))


    _, vel_x = psim.SliderFloat("Velocity X", vel_x, v_min=-3, v_max=3) 
    _, vel_y = psim.SliderFloat("Velocity Z", vel_y, v_min=-5, v_max=5) 
    _, step_size = psim.SliderFloat("Step Size", int(step_size), v_min=1, v_max=20) 
    step_size = int(step_size)

    if(psim.Button("Poke")):
        
        state = state.reshape(-1,6)
        state[:,3] += vel_x / 50.
        state[:,5] += vel_y / 50.
        state = state.reshape(-1)
    

    state = predict(state, step_size)


ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.load_static_material("duck", "data/duck/ceramic_yellow.hdr")
#ps.set_environment_map("studio") 
ps.set_user_callback(callback)
ps.show()




