"""
Dataset classes for fluorescence trace data.
"""
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os

def ema_standardize(x, alpha=0.01, eps=1e-6, clip_val=6.0):
    x = np.asarray(x, dtype=np.float32)
    m, v = 0.0, 1.0
    out = np.empty_like(x)
    for t, xt in enumerate(x):
        m = alpha * xt + (1 - alpha) * m
        # 用偏差平方的 EMA 估计方差
        v = alpha * (xt - m) ** 2 + (1 - alpha) * v
        out[t] = (xt - m) / (np.sqrt(v) + eps)
    if clip_val is not None:
        out = np.clip(out, -clip_val, clip_val)
    return out

class FluorescenceDataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            data_path (string): Path to the npy file with fluorescence data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fluorescence_path = os.path.join(data_path, 'fluorescence_traces.npy')
        self.promoter_path = os.path.join(data_path, 'promoter_states.npy')
        self.fluorescence = np.load(self.fluorescence_path, allow_pickle=True)
        self.promoter = np.load(self.promoter_path, allow_pickle=True)

    def __len__(self):
        return len(self.fluorescence)

    def __getitem__(self, idx):
        fluorescence_holder = self.fluorescence[idx]
        fluorescence_holder = ema_standardize(fluorescence_holder, alpha=0.01, eps=1e-6, clip_val=6.0)
        promoter_holder = self.promoter[idx]

        # Convert to torch tensor
        fluorescence_holder = torch.tensor(fluorescence_holder, dtype=torch.float32)
        promoter_holder = torch.tensor(promoter_holder, dtype=torch.float32)

        return fluorescence_holder, promoter_holder

if __name__ == "__main__":
    # Test the dataset class
    dataset_path = '../dataset/train/'
    fluorescence_dataset = FluorescenceDataset(data_path=dataset_path)
    dataloader = DataLoader(fluorescence_dataset, batch_size=4, shuffle=True, num_workers=0)
    for i, (data, target) in enumerate(dataloader):
        print(f"Batch {i}:")
        print("Data shape:", data.shape)
        print("Target shape:", target.shape)
        break
