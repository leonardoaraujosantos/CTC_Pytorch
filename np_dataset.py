from torch.utils.data import Dataset
import numpy as np

class NPDataset(Dataset):
    def __init__(self, NP_X, NP_Y, transform=None):
    
        self.X = NP_X
        self.Y = NP_Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx): 
        X = self.X[idx]
        Y = self.Y[idx]
        sample = {'X': X, 'Y': Y}

        # Handle Augmentations
        if self.transform:
            sample = self.transform(sample)

        return sample