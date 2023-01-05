import numpy as np, pandas as pd
import os, torch, neurokit2 as nk
from scipy.signal import resample
from torch.utils import data
from tqdm import tqdm


class TILES_ECG(data.Dataset):
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.root = root
        self.participants = split
        self.transform = transform
        self.samples = []

        print("\nLoading participant data ...")
        for prcp in tqdm(os.listdir(self.root)):

            # check if assigned to this split
            if prcp not in self.participants:
                continue

            # load ECG data of participant
            data = np.load(self.root + prcp)
            for s in data:
                self.samples.append(torch.tensor(s.copy()))

        print(f"Loaded {len(self.samples)} ECG samples in total.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        ecg = self.samples[index]
        return self.transform(ecg)


if __name__ == "__main__":
    prcp_list = [
        "02b7a595-6508-46bd-8239-6deb433d6290.npy",
        "13c66354-c2ce-4471-974d-0fd776a8a1bb.npy"
    ]
    dataset = TILES_ECG(root="data/inp/", split=prcp_list)
