import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from torch.nn.functional import avg_pool1d

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
            
        # 移動平均のウィンドウサイズ
        self.moving_avg_window = 5

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        
        # ベースライン補正を追加
        baseline = torch.mean(X[:, :int(X.shape[1] * 0.1)], dim=1, keepdim=True)
        X = X - baseline
        
        # 移動平均フィルタの適用
        X = X.unsqueeze(0)  # (1, channels, time)
        X = avg_pool1d(X, kernel_size=self.moving_avg_window, stride=1, padding=self.moving_avg_window//2)
        X = X.squeeze(0)  # (channels, time)
        
        # パディングによる長さの変化を元に戻す
        X = X[:, :self.X.shape[2]]

        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]