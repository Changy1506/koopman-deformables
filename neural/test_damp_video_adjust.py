import torch
import torch.nn as nn
import h5py
import numpy as np
from eigenvalue_network import eigenvalue_network
from basis_network import basis_network
from data_utils import load_checkpoint, load_snapshots, has_dataset
import sys
import polyscope as ps
import polyscope.imgui as psim
import time
import igl
g = 1
r = 5
eigenvalue_model = eigenvalue_network(g, r)
basis_model = basis_network(g, r)





xx = np.load("data/damp_video/rest_pos_2.npy").reshape(1,-1,2)  
xx = np.concatenate([xx, np.zeros((*xx.shape[:-1], 1), xx.dtype)], axis=-1)


# The ground-truth tracks are only drawn when the dataset is available; otherwise
# the rollout starts from the first frame stored with the pretrained weights.
seq = 2
pretrained = load_checkpoint("damp_video")
num_frames = pretrained["num_frames"][seq]
show_gt = has_dataset("damp_video", "X_%d.pt" % seq)
if show_gt:
    X = load_snapshots("damp_video", "X_%d.pt" % seq).transpose(1,0)  / 10.
else:
    X = pretrained["init_states"][seq].numpy().reshape(1,-1)  / 10.
X = X.reshape(X.shape[0], -1, 2)

X = np.concatenate([X, np.zeros((*X.shape[:-1], 1), X.dtype)], axis=-1)

X = X.reshape(X.shape[0], -1)


x0 = xx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

eigenvalue_model = eigenvalue_model.to(device)
basis_model = basis_model.to(device)

# pretrained weights by default, or pass a checkpoint from training_damp_video.py
ckpt = load_checkpoint("damp_video", sys.argv[1] if len(sys.argv) > 1 else None, map_location=device)
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


print(eigenvalue_normalized)

geo_expand = geo.view(geo.shape[0], 1, geo.shape[-1]).repeat(1, x0.shape[1], 1)
basis_input = torch.cat((x0, geo_expand), dim=-1)
basis = basis_model(basis_input).reshape(eigenvalue.shape[0], x0.shape[1] * 6, r)

state = torch.Tensor(X[0]).to(device).reshape(1,-1) * 100






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


def get_surface(current_state):
    global basis_damp, basis
    
    lhs = torch.bmm(basis.transpose(1, 2), basis)

    rhs = torch.bmm(
                basis.transpose(1, 2),
                current_state.reshape(current_state.shape[0], -1, 1).to(torch.complex64)
            )
    xhat_now = torch.linalg.solve(lhs, rhs)
    
    reconstruct_next = torch.bmm(basis_damp, xhat_now)
    return reconstruct_next.reshape(-1, 6).real


vel_x = 0
vel_y = 0
idx = 0

V_damp, F_damp = igl.read_triangle_mesh("data/damp_video/damp.obj")


V_damp_0 = V_damp.copy()
V_damp[:,1] *= -1
V_damp /= 80.
V_damp[:,0] += 0.127
V_damp[:,1] += 0.085


x0_damp = torch.Tensor(V_damp).reshape(1,-1,3).to(device) * 1000
geo = torch.zeros(1,1).to(device)
geo_expand = geo.view(geo.shape[0], 1, geo.shape[-1]).repeat(1, x0_damp.shape[1], 1)
basis_input_damp = torch.cat((x0_damp, geo_expand), dim=-1)
basis_damp = basis_model(basis_input_damp).reshape(eigenvalue.shape[0], x0_damp.shape[1] * 6, r)


ps.init()


def callback():
    global state
    global vel_x, vel_y, idx, V_damp, F_damp

    

    


    state_now = state.reshape(-1,6).detach().cpu().numpy()
    
    mesh = ps.register_point_cloud("damp_point_cloud", state_now[:,:3] / 100. + xx.reshape(-1,3) / 10)

    if show_gt:
        mesh = ps.register_point_cloud("GT", X[idx % X.shape[0]].reshape(-1,6)[:,:3] + xx.reshape(-1,3) / 10)

        mesh.add_vector_quantity("vel",X[idx % X.shape[0]].reshape(-1,6)[:,3:])




    idx = idx + 1
    if idx >= num_frames:
        idx = 0
        state = torch.Tensor(X[0]).to(device).reshape(1,-1) * 100

    time.sleep(0.05)

    

    


    
    offset_damp = get_surface(state)[:,:3]
    ps.register_surface_mesh("damp! damp!", V_damp + offset_damp.detach().cpu().numpy() / 100., F_damp)
    

    state = predict(state)



ps.set_ground_plane_mode("shadow_only")
ps.set_user_callback(callback)
ps.show()
            


