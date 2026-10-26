import sys
import numpy as np
import h5py
import polyscope as ps
from scipy.linalg import pinv
import polyscope.imgui as psim
import taichi as ti
from scipy.optimize import nnls
from interactive import InteractiveApp
from dmd_utils import load_pretrained


ti.init(arch=ti.cpu)



with h5py.File("data/robot_finger/finger.h5", "r") as f:
    x = f["x"][:]
    f = f["faces"][:]
print(x.shape, f.shape)


# modes / eigenvalues from fit_dmd.py (optionally pass another .npz path)
modes, eigs, _ = load_pretrained("robot_finger", sys.argv[1] if len(sys.argv) > 1 else None)

state = np.zeros(modes.shape[0], dtype=np.float32)
state_0 = state.copy()

mode_inv = pinv(modes)

print(eigs)


def predict(X):
    global modes, mode_inv, eigs
    return np.linalg.multi_dot(
            [modes, np.diag(eigs), mode_inv, X]
        )

vel_x = 0
vel_y = 0
vel_z = 0


@ti.kernel
def apply_force(
    particle_x: ti.types.ndarray(),
    particle_v: ti.types.ndarray(),
    faces: ti.types.ndarray(),   # (F, 3) int
    scale: ti.f32,               # pressure coefficient (can include dt)
):
    for f in range(faces.shape[0]):
        i0 = faces[f, 0]
        i1 = faces[f, 1]
        i2 = faces[f, 2]

        x0 = ti.Vector([particle_x[i0, 0], particle_x[i0, 1], particle_x[i0, 2]])
        x1 = ti.Vector([particle_x[i1, 0], particle_x[i1, 1], particle_x[i1, 2]])
        x2 = ti.Vector([particle_x[i2, 0], particle_x[i2, 1], particle_x[i2, 2]])

        # Unnormalized normal = 2*area * n_hat
        n = (x1 - x0).cross(x2 - x0)
        n_len = n.norm() + 1e-12

        area = 0.5 * n_len
        n_hat = n / n_len

        # Pressure-like force proportional to area, along normal
        F = scale * area * n_hat * 1500 / 500. * 5

        # Distribute equally to the 3 vertices (atomic because shared vertices)
        ti.atomic_add(particle_v[i0, 3], F[0] / 3.0)
        ti.atomic_add(particle_v[i0, 4], F[1] / 3.0)
        ti.atomic_add(particle_v[i0, 5], F[2] / 3.0)
        
        ti.atomic_add(particle_v[i1, 3], F[0] / 3.0)
        ti.atomic_add(particle_v[i1, 4], F[1] / 3.0)
        ti.atomic_add(particle_v[i1, 5], F[2] / 3.0)
        
        ti.atomic_add(particle_v[i2, 3], F[0] / 3.0)
        ti.atomic_add(particle_v[i2, 4], F[1] / 3.0)
        ti.atomic_add(particle_v[i2, 5], F[2] / 3.0)


def affine_to_linear_step(eigs, B, X):
    eigs = np.asarray(eigs)
    B = np.asarray(B)
    X = np.asarray(X)

    n = eigs.shape[0]

    A_prime = np.zeros((n + 1, n + 1), dtype=complex)
    A_prime[:n, :n] = np.diag(eigs)
    A_prime[:n, n] = B
    A_prime[n, n] = 1.0

    X_aug = np.zeros(n + 1, dtype=complex)
    X_aug[:n] = X
    X_aug[n] = 1.0

    X_next_aug = A_prime @ X_aug

    return X_next_aug, A_prime


def diag_geom_sum_times_B(lam, B, t, atol=1e-12):
    lam = np.asarray(lam)
    B = np.asarray(B)
    t = int(t)

    out = np.empty_like(B, dtype=np.result_type(lam, B, np.complex128))

    mask = np.isclose(lam, 1.0, atol=atol)
    out[mask] = (t + 1) * B[mask]

    lm = lam[~mask]
    if lm.size:
        out[~mask] = B[~mask] * (1 - lm**(t + 1)) / (1 - lm)

    return out


def diag_geom_sum(lam, t, atol=1e-12):
    lam = np.asarray(lam)
    t = int(t)

    out = np.empty_like(lam, dtype=np.result_type(lam, np.complex128))

    mask = np.isclose(lam, 1.0, atol=atol)
    out[mask] = (t + 1)

    lm = lam[~mask]
    if lm.size:
        out[~mask] = (1 - lm**(t + 1)) / (1 - lm)

    return out




def get_force(state, loading1, loading2, loading3):
    global face_idx_1, face_idx_2, face_idx_3, x
    global mode_inv

    state = np.ascontiguousarray(state.reshape(-1,6).real)

    disp = np.ascontiguousarray(state[:,:3] + x)

    force = np.zeros(state.shape)

    apply_force(disp, force, face_idx_1, loading1)
    apply_force(disp, force, face_idx_2, loading2)
    apply_force(disp, force, face_idx_3, loading3)

    return mode_inv @ force.reshape(-1)



def solve_system(idx, goal, state):
    global face_idx_1, face_idx_2, face_idx_3, x
    global mode_inv, modes, eigs

    state = state.reshape(-1)

    vel_modes = modes.reshape(-1,6,modes.shape[-1])[:,3:,:].reshape(-1, modes.shape[-1])
    constraint_modes = modes.reshape(-1,6,modes.shape[-1])[idx,np.array([0,2], dtype = np.int64),:].reshape(-1, modes.shape[-1]) * 10000.

    modes_all = np.concatenate((vel_modes, constraint_modes), axis=0)

    vel_goal = np.zeros(vel_modes.shape[0])
    constraint_goal = goal[np.array([0,2], dtype = np.int64)].reshape(-1) * 10000.
    rhs_all = np.concatenate((vel_goal, constraint_goal), axis=0)

    eigs_loading = diag_geom_sum(eigs, 100)
    diag = np.diag(eigs_loading)

    for idx in range(5):

        force_1 = get_force(state, 1, 0, 0)
        force_2 = get_force(state, 0, 1, 0)
        force_3 = get_force(state, 0, 0, 1)

        contribution_part_1 = modes_all @ diag @ force_1
        contribution_part_2 = modes_all @ diag @ force_2
        contribution_part_3 = modes_all @ diag @ force_3

        lhs_all = np.concatenate((contribution_part_1.reshape(-1,1), \
                                  contribution_part_2.reshape(-1,1), \
                                  contribution_part_3.reshape(-1,1)), axis=1)

        #print(lhs_all.shape, rhs_all.shape)

        solution = nnls(lhs_all, rhs_all)[0].reshape(-1)

        reduced_coordinate = diag @ force_1 * solution[0] \
                            + diag @ force_2 * solution[1] \
                            + diag @ force_3 * solution[2]

        state = modes @ reduced_coordinate
        state = np.ascontiguousarray(state.reshape(-1))
    
    return solution


ps.init()

face_idx_1 = np.load('data/robot_finger/idx1.npy')
face_idx_2 = np.load('data/robot_finger/idx2.npy')
face_idx_3 = np.load('data/robot_finger/idx3.npy')

goal_point = np.array([[1.2,2,0.35], [-1.,2,0.3], [-0.7,2,-0.7]])

idx = np.array([1218], dtype=np.int64)

# solutions = []
for i in range(3):
     solution = solve_system(idx, goal_point[i], state) 
#     print(solution)
#     solutions.append(solution)


#ps.register_point_cloud("sel", x)

idx = 0
idx_loading = 0
idx_rendering = 0

app = InteractiveApp()

def simulation():
    global state, dmd, idx
    global vel_x, vel_y, vel_z
    global face_idx_1, face_idx_2, face_idx_3, solutions
    global idx, idx_loading, idx_rendering

    state_now = state.reshape(-1,6)
    mesh = ps.register_surface_mesh("finger", state_now[:,:3] + x, f - 1)

    idx = idx + 1
    idx_rendering = idx_rendering + 1

    state = predict(state).real

    if True:
        
        state = np.ascontiguousarray(state.reshape(-1,6).real)

        disp = np.ascontiguousarray(state[:,:3] + x)
        force = np.zeros(state.shape)

        apply_force(disp, force, face_idx_1, vel_x * 1)
        apply_force(disp, force, face_idx_2, vel_y * 1)
        apply_force(disp, force, face_idx_3, vel_z * 1)


        state = state + force
        state = state.reshape(-1)


current_idx = 0
current_state = 0
    
def callback():
    global app, state_0
    global vel_x, vel_y, vel_z # loadings
    global current_idx, current_state

    app.draw_ui_callback()
    if len(app.polyline_points) > 0 and current_state == 0 and current_idx < len(app.polyline_points):
        # calculate next loading
        goal_point = [app.polyline_3d_scale * app.polyline_points[current_idx][0] - 2, 
                      app.polyline_y_3d, 
                      app.polyline_3d_scale * app.polyline_points[current_idx][1] - 1.5]
        print(goal_point)
        
        idx = np.array([1218], dtype=np.int64)
        solution = solve_system(idx, np.array(goal_point), state_0) 

        vel_x = solution[0]
        vel_y = solution[1]
        vel_z = solution[2]

        current_idx = current_idx + 1
        current_state = 150
    
    simulation()
    current_state = current_state - 1
    if(current_state < 0):
        current_state = 0





ps.set_ground_plane_mode("shadow_only")
ps.look_at((8,6,8), (0,0,0))
ps.set_user_callback(callback)
ps.show()