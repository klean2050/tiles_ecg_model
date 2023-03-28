import numpy as np, os, torch
from ecg_augmentations import *
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

    def generate_label(self, x):
        # signal transforms
        transforms = {
            "0": Pass(),
            "1": PRMask(sr=100),
            "2": QRSMask(sr=100),
            "3": Permute(),
            "4": Scale(),
            "5": TimeWarp(sr=100),
            "6": Invert(),
            "7": Reverse(),
        }
        choice = np.random.choice(
            np.arange(8),
            size=4,
            p=[0.05, 0.2, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05]
        )
        y = np.zeros(8)

        if 0 in choice:
            y[0] = 1
            return x.unsqueeze(0), y
        else:
            y[choice] = 1
            this_transform = [
                tr for k, tr in transforms.items() if int(k) in choice
            ]
            this_transform = ComposeMany(this_transform, 1)
            augmented_x = this_transform(x.unsqueeze(0))
            return augmented_x[0], y

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.transform.num_augmented_samples == 2:
            ecg = self.samples[index].unsqueeze(0)
            return self.transform(ecg)[:, 0]
        else:
            ecg = self.samples[index]
            ecg, y = self.generate_label(ecg)
            # apply standard transforms now
            return self.transform(ecg)[0], y



if __name__ == "__main__":
    prcp_list = [
        "02b7a595-6508-46bd-8239-6deb433d6290.npy",
        "13c66354-c2ce-4471-974d-0fd776a8a1bb.npy",
    ]
    # create transform for ECG augmentation
    transforms = ComposeMany([RandomCrop(n_samples=1000)], 2)
    dataset = TILES_ECG(root="data/tiles/", split=prcp_list, transform=transforms)
    print(dataset[0].shape)
