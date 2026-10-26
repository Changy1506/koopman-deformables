import torch
import torch.nn as nn
import h5py
import numpy as np
from eigenvalue_network import eigenvalue_network
from basis_network import basis_network
from dataloader import DatasetDMD_seqs
from data_utils import load_snapshots, checkpoint_path
from torch.utils.data import Dataset, DataLoader

g = 1
r = 5
eigenvalue_model = eigenvalue_network(g, r)
basis_model = basis_network(g, r)
opt_eig = torch.optim.Adam(eigenvalue_model.parameters(), lr=1e-2)
opt_basis = torch.optim.Adam(basis_model.parameters(), lr=1e-3)


filenames = []
geos = []
xs = []
X = []
Y = []

for filename in [1, 2]:
    
    
    geo_i = [0.]
    geo_i = np.array(geo_i).reshape(1,-1)
    
    x_i = np.load("data/damp_video/rest_pos_" + str(filename) + ".npy").reshape(1,-1,2) 
    x_i = np.concatenate([x_i, np.zeros((*x_i.shape[:-1], 1), x_i.dtype)], axis=-1) 

        
    X_i = load_snapshots("damp_video", "X_" + str(filename) + ".pt").transpose(1,0) / 10.
    Y_i = load_snapshots("damp_video", "Y_" + str(filename) + ".pt").transpose(1,0) / 10.

    X_i = X_i.reshape(X_i.shape[0], -1, 2)
    Y_i = Y_i.reshape(Y_i.shape[0], -1, 2)

    X_i = np.concatenate([X_i, np.zeros((*X_i.shape[:-1], 1), X_i.dtype)], axis=-1)
    Y_i = np.concatenate([Y_i, np.zeros((*Y_i.shape[:-1], 1), Y_i.dtype)], axis=-1)

    print(X_i.shape, Y_i.shape)

    X_i = X_i.reshape(X_i.shape[0], -1)
    Y_i = Y_i.reshape(Y_i.shape[0], -1)

    X.append(X_i)
    Y.append(Y_i)
    
    geo = np.repeat(geo_i, X_i.shape[0], axis=0)
    geos.append(geo)

    x0_i = np.repeat(x_i, X_i.shape[0], axis=0)
    xs.append(x0_i)


print(len(X))

dataset = DatasetDMD_seqs(X, Y, geos, xs)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


eigenvalue_model = eigenvalue_model.to(device)
basis_model = basis_model.to(device)
#print(x0)

def train():
    eigenvalue_model.train()
    basis_model.train()

    for epoch in range(500):
        for current_state, next_state, geo, x0 in loader:

            current_state = current_state.to(device, non_blocking=True)
            next_state    = next_state.to(device, non_blocking=True)
            geo           = geo.to(device, non_blocking=True)
            x0            = x0.to(device, non_blocking=True)

            #print(current_state.shape, next_state.shape, geo.shape, x0.shape)

            

            eigenvalue_input = geo
            eigenvalue = eigenvalue_model(eigenvalue_input)
            eps = 1e-8
           
            norm = eigenvalue.abs()

            eigenvalue_normalized = torch.where(
                norm > 1.0,
                eigenvalue / (norm + eps),
                eigenvalue)

            geo_expand = geo.view(geo.shape[0], 1, geo.shape[-1]).repeat(1, x0.shape[1], 1)
            basis_input = torch.cat((x0, geo_expand), dim=-1)

            #print(basis_input.shape)

            basis = basis_model(basis_input).reshape(eigenvalue.shape[0], x0.shape[1] * 6, r)

            #print(basis)

            #print(basis.shape)

            lhs = torch.bmm(basis.transpose(1, 2), basis)

            rhs = torch.bmm(
                basis.transpose(1, 2),
                current_state.reshape(current_state.shape[0], -1, 1).to(torch.complex64)
            )
            xhat_now = torch.linalg.solve(lhs, rhs)

            rhs_next = torch.bmm(
                basis.transpose(1, 2),
                next_state.reshape(next_state.shape[0], -1, 1).to(torch.complex64)
            )
            xhat_next = torch.linalg.solve(lhs, rhs_next)

            reconstruct_current = torch.bmm(basis, xhat_now)
            reconstruct_loss = torch.mean((reconstruct_current.reshape_as(current_state).real - current_state) ** 2) \
                               #+ torch.mean((reconstruct_current.reshape_as(current_state).imag) ** 2)

            xhat_pred_next = torch.bmm(torch.diag_embed(eigenvalue_normalized), xhat_now)
            loss_reduced = torch.sum((xhat_pred_next.real - xhat_next.real) ** 2) + torch.sum((xhat_pred_next.imag - xhat_next.imag) ** 2)
            #loss_reduced = loss_reduced / (torch.linalg.norm(xhat_next.abs()) + eps) * 100.

            reconstruct_next = torch.bmm(basis, xhat_pred_next)
            reconstruct_loss_next = torch.mean((reconstruct_next.reshape_as(next_state).real - next_state) ** 2) \
                                    #+ torch.mean((reconstruct_next.reshape_as(next_state).imag) ** 2)
            loss = reconstruct_loss * 100. + loss_reduced * 10000. + reconstruct_loss_next * 10000.

            opt_eig.zero_grad()
            opt_basis.zero_grad()

            print(f"epoch {epoch} | loss {loss.item():.6e}  | loss_reduced {loss_reduced.item():.6e}  | reconstruct_loss {reconstruct_loss.item():.6e} | reconstruct_loss_next {reconstruct_loss_next.item():.6e}")

            loss.backward()
            opt_eig.step()
            opt_basis.step()

    torch.save(
        {
            "eigenvalue_model": eigenvalue_model.state_dict(),
            "basis_model": basis_model.state_dict(),
            "g": g,
            "r": r,
        },
        checkpoint_path("damp_video")
    )
    print("Saved to", checkpoint_path("damp_video"))


train()