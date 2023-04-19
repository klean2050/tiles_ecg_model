import os, numpy as np
import pandas as pd, neurokit2 as nk
from random import shuffle
from tqdm import tqdm
from torch.utils import data
from scipy.signal import resample_poly


class EPIC(data.Dataset):
    def __init__(self, root, sr, scenario=1, split="train", category="arousal"):
        super().__init__()
        self.sr = sr
        self.win = sr * 10

        # sanity checks
        assert scenario in [1, 2, 3, 4], "Scenario must be in [1, 2, 3, 4]"
        if scenario == 1:
            assert split in [
                "train",
                "dev",
                "test",
            ], "Split must be in [train, dev, test]"
            self.split = split if split != "dev" else "train"
        else:
            assert split in [0, 1, 2, 3, 4], "Split must be in [0, 1, 2, 3, 4]"
            self.split = f"fold_{split}"

        self.root = root + f"/scenario_{scenario}/{self.split}/"
        if os.path.exists(f"data/epic/{scenario}_{self.split}_ecg.npy"):
            print("Loading from cache...")
            ecg_data = np.load(f"data/epic/{scenario}_{self.split}_ecg.npy")
            if os.path.exists(f"data/epic/{scenario}_{self.split}_{category}.npy"):
                ecg_labels = np.load(
                    f"data/epic/{scenario}_{self.split}_{category}.npy"
                )
            else:
                ecg_labels = list()
                for csv_file in tqdm(os.listdir(self.root + "annotations")):
                    label_path = os.path.join(self.root, "annotations", csv_file)
                    labels = pd.read_csv(label_path)
                    label_values = labels[category].values
                    label_times = labels["time"].values
                    for i, time in enumerate(label_times):
                        # annotation every 50 ms
                        if time < 10000:
                            continue  # skip first 10 seconds
                        ecg_labels.append(label_values[i])
                ecg_labels = np.array(ecg_labels)
                np.save(f"data/epic/{scenario}_{self.split}_{category}.npy", ecg_labels)

            # handle metadata
            names = list()
            for csv_file in tqdm(os.listdir(self.root + "annotations")):
                label_path = os.path.join(self.root, "annotations", csv_file)
                labels = pd.read_csv(label_path)
                label_times = labels["time"].values
                for i, time in enumerate(label_times):
                    # annotation every 50 ms
                    if time < 10000:
                        continue
                    names.append(csv_file[:-4])
            names = np.array(names)
        else:
            print("Loading ECG data...")
            ecg_data, ecg_labels = list(), list()
            for csv_file in tqdm(os.listdir(self.root + "physiology")):

                label_path = os.path.join(self.root, "annotations", csv_file)
                ecg_path = os.path.join(self.root, "physiology", csv_file)

                # Read ECG data
                ecg = pd.read_csv(ecg_path)["ecg"].values
                new_len = int((len(ecg) / 1000) * self.sr)
                ecg = resample_poly(ecg, self.sr, 1000)[:new_len]

                # Read data and labels
                labels = pd.read_csv(label_path)
                label_values = labels[category].values
                label_times = labels["time"].values
                for i, time in enumerate(label_times):
                    # annotation every 50 ms
                    if time < 10000:
                        continue  # skip first 10 seconds

                    # divide by 10 == multiply by 100 and divide by 1000
                    start_idx = int(time / 10 - self.win)
                    end_idx = int(time / 10)
                    data = ecg[start_idx:end_idx]

                    # ECG data cleaning
                    data = nk.ecg_clean(data, sampling_rate=self.sr)
                    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
                    data = (data - mean) / (std + 1e-5)

                    ecg_data.append(data)
                    ecg_labels.append(label_values[i])

            ecg_data = np.array(ecg_data)
            ecg_labels = np.array(ecg_labels)

            os.makedirs("data/epic", exist_ok=True)
            np.save(f"data/epic/{scenario}_{self.split}_ecg.npy", ecg_data)
            np.save(f"data/epic/{scenario}_{self.split}_{category}.npy", ecg_labels)

        # partition validation set
        if scenario == 1 and split != "test":
            val_indices = list()
            i = 0
            while i < len(names):
                # identify all samples with the same name
                indices = np.where(names == names[i])[0]
                # isolate 10% of the samples as validation set
                num_val_samples = int(len(indices) * 0.1)
                # select 10% of the samples in the end
                val_indices.append(indices[-num_val_samples:])
                i += len(indices)
            val_indices = np.concatenate(val_indices)

            if split == "dev":
                ecg_data = ecg_data[val_indices]
                ecg_labels = ecg_labels[val_indices]
            else:
                ecg_data = np.delete(ecg_data, val_indices, axis=0)
                ecg_labels = np.delete(ecg_labels, val_indices, axis=0)

        # low data regime
        self.samples = ecg_data[::1] if split == "train" else ecg_data
        self.labels = ecg_labels[::1] if split == "train" else ecg_labels
        print(f"Loaded {len(self.labels)} ECG samples in total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecg = self.samples[idx]
        lab = self.labels[idx]
        return ecg, lab, 0


if __name__ == "__main__":
    root = "/home/kavra/Datasets/physio/epic_challenge/"
    train_dataset = EPIC(root, sr=100, scenario=1, split="train", category="valence")
    test_dataset = EPIC(root, sr=100, scenario=1, split="dev", category="valence")
    print(train_dataset[0][0].shape, train_dataset[0][1])
