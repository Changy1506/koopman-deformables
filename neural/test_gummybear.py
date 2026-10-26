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

g = 1
r = 20
eigenvalue_model = eigenvalue_network(g, r)
basis_model = basis_network(g, r)





with h5py.File("data/gummybear/gummybear.h5", "r") as f:
    xx = f["x"][:].reshape(1,-1,3)
    ff = f["faces"][:]

x0 = xx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

eigenvalue_model = eigenvalue_model.to(device)
basis_model = basis_model.to(device)

# pretrained weights by default, or pass a checkpoint from training_gummybear.py
ckpt = load_checkpoint("gummybear", sys.argv[1] if len(sys.argv) > 1 else None, map_location=device)
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

    #print(eigenvalue_normalized)
    
    xhat_pred_next = torch.bmm(torch.diag_embed(eigenvalue_normalized), xhat_now)
            
    reconstruct_next = torch.bmm(basis, xhat_pred_next)
    #print(reconstruct_next.shape)
    return reconstruct_next.reshape_as(current_state).real


vel_x = 0
def callback():
    global state
    global vel_x

    state_now = state.reshape(-1,6).detach().cpu().numpy()
    
    mesh = ps.register_surface_mesh("gummybear", state_now[:,:3] / 100. + xx.reshape(-1,3), ff - 1)
    mesh.add_vector_quantity("vel",basis[:,:,0].reshape(-1,6)[:,3:].detach().cpu().numpy())

    _, vel_x = psim.SliderFloat("vel_x", vel_x, v_min=-0.5, v_max=0.5) 
    
    if(psim.Button("Poke")):
        
        state = state.reshape(-1,6)
        state[:,3] += vel_x * 5
        state = state.reshape(1, -1)


    state = predict(state)


ps.init()
ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)
ps.show()
            


