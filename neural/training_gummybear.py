import torch
import torch.nn as nn
import h5py
import numpy as np
from eigenvalue_network import eigenvalue_network
from basis_network import basis_network
from dataloader import DatasetDMD
from data_utils import load_snapshots, checkpoint_path
from torch.utils.data import Dataset, DataLoader

g = 1
r = 20
eigenvalue_model = eigenvalue_network(g, r)
basis_model = basis_network(g, r)
opt_eig = torch.optim.Adam(eigenvalue_model.parameters(), lr=1e-2)
opt_basis = torch.optim.Adam(basis_model.parameters(), lr=1e-2)



with h5py.File("data/gummybear/gummybear.h5", "r") as f:
    x = f["x"][:].reshape(1,-1,3)
print(x.shape)

X = load_snapshots("gummybear", "X.pt").transpose(1,0)
Y = load_snapshots("gummybear", "Y.pt").transpose(1,0)

print(X.shape)
geo = np.zeros((X.shape[0], 1))
x0 = np.repeat(x, X.shape[0], axis=0)
dataset = DatasetDMD(X, Y, geo, x0)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ckpt = torch.load("duck_dmd_models.pt", map_location=device)
# eigenvalue_model.load_state_dict(ckpt["eigenvalue_model"])
# basis_model.load_state_dict(ckpt["basis_model"])

eigenvalue_model = eigenvalue_model.to(device)
basis_model = basis_model.to(device)
#print(x0)

def train():
    eigenvalue_model.train()
    basis_model.train()

    for epoch in range(1000):
        for current_state, next_state, geo, x0 in loader:

            current_state = current_state.to(device, non_blocking=True)
            next_state    = next_state.to(device, non_blocking=True)
            geo           = geo.to(device, non_blocking=True)
            x0            = x0.to(device, non_blocking=True)

            

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
            loss = reconstruct_loss * 1000. + loss_reduced * 10000. + reconstruct_loss_next * 10000.

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
        checkpoint_path("gummybear")
    )
    print("Saved to", checkpoint_path("gummybear"))

train()