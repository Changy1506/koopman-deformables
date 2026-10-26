import torch
from torch.utils.data import Dataset, DataLoader

class DatasetDMD(Dataset):
    def __init__(self, X, Y, geo, x0):
        
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
        self.geo = torch.as_tensor(geo, dtype=torch.float32)
        self.x0 = torch.as_tensor(x0, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx] * 100, self.Y[idx] * 100, self.geo[idx], self.x0[idx] * 100


class DatasetDMD_seqs(Dataset):
    def __init__(self, X, Y, geo, x0):
        
        self.X = []
        for X_i in X:
            self.X.append(torch.as_tensor(X_i, dtype=torch.float32))
        
        self.Y = []
        for Y_i in Y:
            self.Y.append(torch.as_tensor(Y_i, dtype=torch.float32))
        
        self.geo = []
        for geo_i in geo:
            self.geo.append(torch.as_tensor(geo_i, dtype=torch.float32))
        
        self.x0 = []
        for x0_i in x0:
            self.x0.append(torch.as_tensor(x0_i, dtype=torch.float32))
        
        self.len = 0
        for X_i in X:
            self.len += X_i.shape[0]


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        seq = 0
        prefix = 0
        for X_i in self.X:
            if prefix <= idx and (seq == len(self.X) - 1 or prefix + X_i.shape[0] > idx):
                offset = idx - prefix
                break
            prefix += X_i.shape[0]
            seq = seq + 1
        #print(seq, prefix, idx)
        return self.X[seq][offset] * 100, self.Y[seq][offset] * 100, self.geo[seq][offset], self.x0[seq][offset] * 100