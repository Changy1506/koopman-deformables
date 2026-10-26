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
import meshio
import igl

g = 1
r = 20
eigenvalue_model = eigenvalue_network(g, r)
basis_model = basis_network(g, r)


# with h5py.File("data/duck/duck.h5", "r") as f:
#     xx = f["x"][:].reshape(1,-1,3)
#     ff = f["faces"][:]


# the basis network is queried on a different tet mesh than the one used for training
# (data/duck/duck_{2.5k,5k,10k,50k}.mesh)
mesh = meshio.read("data/duck/duck_5k.mesh")
tetmesh = mesh.cells_dict.get("tetra") 
xx = mesh.points.reshape(1,-1,3)


ff = igl.boundary_facets(tetmesh)
if isinstance(ff, tuple):  # libigl >= 2.6 returns (faces, J, K)
    ff = ff[0]

x0 = np.repeat(xx, 1, axis=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

eigenvalue_model = eigenvalue_model.to(device)
basis_model = basis_model.to(device)

# pretrained weights by default, or pass a checkpoint from training_duck.py
ckpt = load_checkpoint("duck", sys.argv[1] if len(sys.argv) > 1 else None, map_location=device)
eigenvalue_model.load_state_dict(ckpt["eigenvalue_model"])
basis_model.load_state_dict(ckpt["basis_model"])
eigenvalue_model.eval()
basis_model.eval()

x0 = torch.Tensor(x0[0]).reshape(1,-1,3).to(device) * 100
geo = torch.zeros(1,1).to(device)
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

def predict(current_state):
    
    lhs = torch.bmm(basis.transpose(1, 2), basis)

    rhs = torch.bmm(
                basis.transpose(1, 2),
                current_state.reshape(current_state.shape[0], -1, 1).to(torch.complex64)
            )
    xhat_now = torch.linalg.solve(lhs, rhs)
    xhat_pred_next = torch.bmm(torch.diag_embed(eigenvalue_normalized), xhat_now)
            
    reconstruct_next = torch.bmm(basis, xhat_pred_next)
    return reconstruct_next.reshape_as(current_state).real


vel_x = 0
vel_y = 0

def callback():
    global state
    global vel_x, vel_y


    state_now = state.reshape(-1,6).detach().cpu().numpy()

    mesh = ps.register_surface_mesh("duck", state_now[:,:3] / 100. + xx.reshape(-1,3), ff)


    mesh.add_vector_quantity("vel",basis[:,:,0].reshape(-1,6)[:,3:].detach().cpu().numpy())

    _, vel_x = psim.SliderFloat("vel_x", vel_x, v_min=-5, v_max=5)
    _, vel_y = psim.SliderFloat("vel_y", vel_y, v_min=-5, v_max=5)

    if(psim.Button("Poke")):

        state = state.reshape(-1,6)
        state[:,3] += vel_x * 5

        state[:,5] += vel_y * 5
        state = state.reshape(1, -1)

    state = predict(state)


ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)
ps.show()
            


