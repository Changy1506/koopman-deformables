import sys
import numpy as np
import h5py
import polyscope as ps
from scipy.linalg import pinv
import polyscope.imgui as psim
import math
import taichi as ti
from dmd_utils import load_pretrained

ti.init(arch=ti.cpu)


with h5py.File("data/bar/bar.h5", "r") as f:
    x = f["x"][:]
    f = f["faces"][:]
print(x.shape, f.shape)


# modes / eigenvalues from fit_dmd.py (optionally pass another .npz path)
modes, eigs, _ = load_pretrained("bar", sys.argv[1] if len(sys.argv) > 1 else None)

state = np.zeros(modes.shape[0], dtype=np.float32)

mode_inv = pinv(modes)

def predict(X, step_size = 1):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs ** step_size), mode_inv, X]
        )

@ti.kernel
def apply_drag(
    particle_x: ti.types.ndarray(),
    particle_v: ti.types.ndarray(),
    particle_x0: ti.types.ndarray(),
    t: float
    ):

    for tid in range(particle_v.shape[0]):

        
        x_0 = particle_x0[tid, 0] 
        y_0 = particle_x0[tid, 1]
        z_0 = particle_x0[tid, 2] 
        if(y_0 < -0.9):
          x_tar = x_0 * ti.cos(t) - z_0 * ti.sin(t) 
          y_tar = y_0
          z_tar = x_0 * ti.sin(t) + z_0 * ti.cos(t) 
          particle_v[tid, 0] = (x_tar - x_0)
          particle_v[tid, 1] = (y_tar - y_0)
          particle_v[tid, 2] = (z_tar - z_0) 
          
        if(y_0 > 0.9):  
          x_tar = x_0 * ti.cos(-t) - z_0 * ti.sin(-t) 
          y_tar = y_0
          z_tar = x_0 * ti.sin(-t) + z_0 * ti.cos(-t) 
          particle_v[tid, 0] = (x_tar - x_0)
          particle_v[tid, 1] = (y_tar - y_0)
          particle_v[tid, 2] = (z_tar - z_0) 
          

vel_x = 0
vel_y = 0
step_size = 1
idx = 0
def callback():
    global state, dmd, idx
    global vel_x, vel_y, step_size
    state_now = np.ascontiguousarray(state.reshape(-1,6).real)
    mesh = ps.register_surface_mesh("Duck", state_now[:,:3] + x, f - 1)
    mesh.set_material("duck")
    mesh.set_color((0.85, 0.65, 0.25))



    
    pos_now = np.ascontiguousarray(state_now[:,:3] + x)

    apply_drag(pos_now, state_now, x, min(math.pi / 2, idx / 20))
    state = state_now.reshape(state.shape)

    state = predict(state, step_size)
    idx = idx + 1


ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.load_static_material("duck", "data/duck/ceramic_yellow.hdr")
#ps.set_environment_map("studio") 
ps.set_user_callback(callback)
ps.show()




