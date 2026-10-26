import torch
import torch.nn as nn
import h5py
import numpy as np
from eigenvalue_network import eigenvalue_network
from basis_network import basis_network
from data_utils import load_checkpoint
import sys
import polyscope as ps
import polyscope.imgui as psim
import taichi as ti
import os


ti.init(arch=ti.cpu)

g = 5
r = 20
eigenvalue_model = eigenvalue_network(g, r)
basis_model = basis_network(g, r)





filename = sorted(os.listdir('data/finger/h5'))[4]
print(filename)
geo_i = [float(x) for x in filename[:-3].split("_")]

print("----------------------- GEOMETRY START --------------------------")

print("gap", (geo_i[0] - 0.04) / (0.08 - 0.04))
print("wall", (geo_i[1] - 0.02) / (0.03 - 0.02))
print("a0", (geo_i[2] - 0.04) / (0.08 - 0.04))
print("amid", (geo_i[3] - 0.04) / (0.08 - 0.04))
print("a1", (geo_i[4] - 0.04) / (0.08 - 0.04))


print("----------------------- GEOMETRY END --------------------------")

geo_i = np.array(geo_i).reshape(1,-1)
with h5py.File("data/finger/h5/" + filename, "r") as f:
     x_i = f["x"][:].T.reshape(1,-1,3)
     ff = f["faces"][:].T
        
geo = geo_i
x0 = x_i

face_idx = np.load('data/finger/npys/'+filename[:-3]+'.npy')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

eigenvalue_model = eigenvalue_model.to(device)
basis_model = basis_model.to(device)

# pretrained weights by default, or pass a checkpoint from training_finger.py
ckpt = load_checkpoint("finger", sys.argv[1] if len(sys.argv) > 1 else None, map_location=device)
eigenvalue_model.load_state_dict(ckpt["eigenvalue_model"])
basis_model.load_state_dict(ckpt["basis_model"])
eigenvalue_model.eval()
basis_model.eval()

x0 = torch.Tensor(x0[0]).reshape(1,-1,3).to(device) * 100
geo = torch.Tensor(geo[0]).reshape(1,-1).to(device)
eigenvalue_input = geo
eigenvalue = eigenvalue_model(eigenvalue_input)
eps = 1e-8
norm = eigenvalue.abs()

eigenvalue_normalized = torch.where(
                norm > 1.0,
                eigenvalue / (norm + eps),
                eigenvalue)

#eigenvalue_normalized *= 0.98
print(eigenvalue_normalized)

geo_expand = geo.view(geo.shape[0], 1, geo.shape[-1]).repeat(1, x0.shape[1], 1)
basis_input = torch.cat((x0, geo_expand), dim=-1)
basis = basis_model(basis_input).reshape(eigenvalue.shape[0], x0.shape[1] * 6, r)

state = torch.zeros(1, x0.shape[1] * 6).to(device)




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
        F = scale * area * n_hat * 3500 / 200. * 500

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

def predict(current_state):
    

    

    lhs = torch.bmm(basis.transpose(1, 2), basis)

    rhs = torch.bmm(
                basis.transpose(1, 2),
                current_state.reshape(current_state.shape[0], -1, 1).to(torch.complex64)
            )
    xhat_now = torch.linalg.solve(lhs, rhs)

    #print(eigenvalue_normalized)
    
    xhat_pred_next = torch.bmm(torch.diag_embed(eigenvalue_normalized), xhat_now)
            
    reconstruct_next = torch.bmm(basis, xhat_pred_next)
    #print(reconstruct_next.shape)
    return reconstruct_next.reshape_as(current_state).real


vel_x = 0
vel_y = 0
poke = -1  # press "Poke" to inflate for 160 frames, then release
def callback():
    global state
    global vel_x, vel_y, poke
    

    state_now = state.reshape(-1,6).detach().cpu().numpy()
    
    mesh = ps.register_surface_mesh("finger", state_now[:,:3] / 100. + x_i.reshape(-1,3), ff - 1)
    mesh.add_vector_quantity("vel",basis[:,:,0].reshape(-1,6)[:,3:].detach().cpu().numpy())

    # _, vel_x = psim.SliderFloat("vel_x", vel_x, v_min=-5, v_max=5) 
    # _, vel_y = psim.SliderFloat("vel_y", vel_y, v_min=-5, v_max=5) 

    if(psim.Button("Poke")):
        poke = 160
    

    if(poke >= 0):
        state = np.ascontiguousarray(state.reshape(-1,6).cpu().detach().numpy())
        apply_force(state_now[:,:3] / 100. + x_i.reshape(-1,3), state, face_idx, 1)
        
        state = torch.Tensor(state).reshape(1, -1).to(device)
    poke = poke - 1


    state = predict(state)


ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)
ps.show()
            


