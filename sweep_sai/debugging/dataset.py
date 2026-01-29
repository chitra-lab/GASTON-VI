import torch
from torch.utils.data import Dataset

class SpatialDataset(Dataset):
    def __init__(self, X, S):
        """
        X: (N, G) tensor
        S: (N, s_dim) tensor
        """
        assert X.size(0) == S.size(0)
        self.X = X
        self.S = S

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return {
            "gene": self.X[idx],
            "spatial": self.S[idx],
        }
