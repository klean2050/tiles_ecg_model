import os, numpy as np
import pandas as pd, neurokit2 as nk
from tqdm import tqdm
from torch.utils import data
from scipy.signal import resample_poly


class EPIC(data.Dataset):
    def __init__(self, root, sr, scenario=1, split="train", category="arousal", fold=0):
        super().__init__()
        self.sr = sr
        self.win = sr * 10
        self.jump = 5
        ecg_data, ecg_labels = list(), list()

        # sanity checks
        assert scenario in [1, 2, 3, 4], "Scenario must be in [1, 2, 3, 4]"
        assert split in ["train", "dev", "test"], "Split must be in [train, dev, test]"
        self.split = split if split != "dev" else "train"

        # setup scenario
        if scenario != 1:
            self.root = root + f"/scenario_{scenario}/fold_{fold}/{self.split}/"
            self.sc = f"{scenario}_{fold}"
        else:
            self.root = root + f"/scenario_{scenario}/{self.split}/"
            self.sc = f"{scenario}"

        if os.path.exists(f"data/epic/{self.sc}_{self.split}_ecg.npy"):
            print("Loading from cache...")
            ecg_data = np.load(f"data/epic/{self.sc}_{self.split}_ecg.npy")
            if self.split == "train":
                if os.path.exists(f"data/epic/{self.sc}_{self.split}_{category}.npy"):
                    ecg_labels = np.load(
                        f"data/epic/{self.sc}_{self.split}_{category}.npy"
                    )
                    names = np.load(f"data/epic/{self.sc}_{self.split}_names.npy")
                else:
                    ecg_labels, names = self.get_labels(category, self.jump)
                    np.save(f"data/epic/{self.sc}_{self.split}_names.npy", names)
                    np.save(
                        f"data/epic/{self.sc}_{self.split}_{category}.npy", ecg_labels
                    )
        else:
            print(f"Loading ECG {split} data...")
            for csv_file in tqdm(os.listdir(self.root + "physiology")):

                label_path = os.path.join(self.root, "annotations", csv_file)
                ecg_path = os.path.join(self.root, "physiology", csv_file)

                # Read ECG data
                ecg = pd.read_csv(ecg_path)["ecg"].values
                new_len = int((len(ecg) / 1000) * self.sr)
                ecg = resample_poly(ecg, self.sr, 1000)[:new_len]

                # Read data and labels
                labels = pd.read_csv(label_path)
                label_times = labels["time"].values
                if self.split == "train":
                    label_times = label_times[:: self.jump]
                for time in label_times:
                    # annotation every 50 ms
                    if time < 10000:
                        continue  # skip first 10 seconds

                    # divide by 10 = (multiply by 100 + divide by 1000)
                    start_idx = int(time / 10 - self.win)
                    end_idx = int(time / 10)
                    data = ecg[start_idx:end_idx]

                    # ECG data cleaning
                    data = nk.ecg_clean(data, sampling_rate=self.sr)
                    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
                    data = (data - mean) / (std + 1e-5)

                    # saving
                    ecg_data.append(data)

            ecg_data = np.array(ecg_data)
            os.makedirs("data/epic", exist_ok=True)
            np.save(f"data/epic/{self.sc}_{self.split}_ecg.npy", ecg_data)

            if self.split == "train":
                ecg_labels, names = self.get_labels(category, jump=self.jump)
                np.save(f"data/epic/{self.sc}_{self.split}_names.npy", names)
                np.save(f"data/epic/{self.sc}_{self.split}_{category}.npy", ecg_labels)

        # partition validation set
        if self.split == "train":
            val_indices = self.define_validation(names, scenario)
            if split == "dev":
                ecg_data = ecg_data[val_indices]
                ecg_labels = ecg_labels[val_indices]
            else:
                ecg_data = np.delete(ecg_data, val_indices, axis=0)
                ecg_labels = np.delete(ecg_labels, val_indices, axis=0)

        # low data regime
        self.samples = ecg_data[::1] if split == "train" else ecg_data
        self.labels = ecg_labels[::1] if split == "train" else ecg_labels
        if split == "test":
            self.labels = np.zeros(len(self.samples))
        print(f"Loaded {len(self.samples)} ECG samples in total.")

    def get_labels(self, category, jump):
        print("Loading labels...")
        ecg_labels, names = list(), list()
        for csv_file in tqdm(os.listdir(self.root + "annotations")):
            label_path = os.path.join(self.root, "annotations", csv_file)
            labels = pd.read_csv(label_path)
            label_values = labels[category].values[::jump]
            label_times = labels["time"].values[::jump]
            for i, time in enumerate(label_times):
                # annotation every 50 ms
                if time < 10000:
                    continue  # skip first 10 seconds
                ecg_labels.append(label_values[i])
                names.append(csv_file[:-4])
        return np.array(ecg_labels), np.array(names)

    def define_validation(self, names, scenario):
        if scenario == 2:
            names = [n.split("_")[1] for n in names]
        elif scenario > 2:
            names = [n.split("_")[3] for n in names]
        names = np.array(names)

        val_indices, i = list(), 0
        while i < len(names):
            # identify all samples with the same name
            indices = np.where(names == names[i])[0]

            if scenario == 1:
                # isolate 10% of the samples as validation set
                num_val_samples = int(len(indices) * 0.1)
                # select 10% of the samples in the end
                selected_indices = indices[-num_val_samples:]
            else:
                # decide if name is in the validation set
                if i < len(names) / 10:
                    selected_indices = indices
                else:
                    break

            val_indices.append(selected_indices)
            # jump to the next session
            i += len(indices)
        return np.concatenate(val_indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecg = self.samples[idx]
        lab = self.labels[idx]
        return ecg, lab, 0


if __name__ == "__main__":
    root = "/home/kavra/Datasets/physio/epic_challenge/"
    ###################################################################
    sample_dataset = EPIC(
        root, sr=100, scenario=3, split="train", category="valence", fold=0
    )
    ###################################################################
    print(sample_dataset[0][0].shape, sample_dataset[0][1])
