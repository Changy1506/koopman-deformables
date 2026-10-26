import sys
import numpy as np
import h5py
import polyscope as ps
from scipy.linalg import pinv
import taichi as ti
from dmd_utils import load_pretrained
ti.init(arch=ti.cpu)


with h5py.File("data/mushroom/mushroom.h5", "r") as f:
    x = f["x"][:]
    f = f["faces"][:]
print(x.shape, f.shape)


# modes / eigenvalues from fit_dmd.py (optionally pass another .npz path)
modes, eigs, _ = load_pretrained("mushroom", sys.argv[1] if len(sys.argv) > 1 else None)

state = np.zeros(modes.shape[0], dtype=np.float32)

mode_inv = pinv(modes)




print(eigs)

@ti.kernel
def apply_force(
    particle_x0: ti.types.ndarray(),
    particle_v: ti.types.ndarray()
):
    for tid in range(particle_v.shape[0]):
        
        x_0 = particle_x0[tid, 0]
        y_0 = particle_x0[tid, 1]
        z_0 = particle_x0[tid, 2]
        if not (y_0 > 0):

            particle_v[tid, 4] = particle_v[tid, 4] - 1 / 500 * 100 * 0.5


def predict(X):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs), mode_inv, X]
        )

def callback():
    global state, dmd
    state_now = np.ascontiguousarray(state.reshape(-1,6).real)
    ps.register_surface_mesh("mushroom", state_now[:,:3] + x, f - 1)

    apply_force(x, state_now)

    state = state_now.reshape(-1)
    

    
    state = predict(state)


ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)
ps.show()




