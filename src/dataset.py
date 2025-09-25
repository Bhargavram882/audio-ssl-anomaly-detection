import numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path

def augment(x):
    # simple time-masking augmentation
    x = x.copy()
    t = np.random.randint(0, x.shape[1]//2)
    x[:, t:t+10] = 0
    return x

class MelPairDataset(Dataset):
    def __init__(self, path="data/mels"):
        self.files = list(Path(path).glob("normal_*.npz"))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        feat = np.load(self.files[idx])["feat"]
        x1 = torch.tensor(augment(feat)).float()
        x2 = torch.tensor(augment(feat)).float()
        return x1.unsqueeze(0), x2.unsqueeze(0)
