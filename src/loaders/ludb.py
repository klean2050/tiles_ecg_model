import wfdb, numpy as np
import os, neurokit2 as nk

from tqdm import tqdm
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class LUDBDataset(Dataset):
    def __init__(self, root):
        self.data_dir = root + "data/"
        self.dat_files = sorted(
            [f for f in os.listdir(self.data_dir) if f.endswith(".dat")]
        )
        print("Loading LUDB data ...")
        self.samples, self.labels = [], []
        for file in tqdm(self.dat_files):
            fpath = os.path.join(self.data_dir, file)
            root = os.path.splitext(fpath)[0]
            samples, rhythm = self.process_prcp(root)
            self.samples.append(samples)
            self.labels.append(rhythm)

        self.samples = np.vstack(self.samples)
        self.names = np.arange(len(self.samples))
        print(f"Loaded {len(self.samples)} ECG samples in total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx], self.names[idx]

    def process_prcp(self, root):
        signals, fields = wfdb.rdsamp(root)
        signal = signals[:, 0]

        new_sig = resample(
            signal, 100 * len(signal) // fields["fs"]
        )  # Downsample to 100 Hz
        sample = nk.ecg_clean(new_sig, sampling_rate=100)
        sample = StandardScaler().fit_transform(sample.reshape(-1, 1))

        # Extract rhythm information from fields
        rhythm_str = fields["comments"][3]
        rhythm = 0 if "Sinus rhythm." in rhythm_str else 1

        return sample[:, 0], rhythm


if __name__ == "__main__":
    path = "/data/avramidi/LUDB/"
    dataset = LUDBDataset(path)
    print(dataset[0][0].shape, dataset[0][1])
