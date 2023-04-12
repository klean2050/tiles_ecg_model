import os, wfdb, torch, numpy as np
import pandas as pd, neurokit2 as nk

from tqdm import tqdm
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import normalize, StandardScaler
from torch.utils.data import Dataset, DataLoader

class LUDBDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.dat')])
        self.samples = []
        for file in self.dat_files:
            dat_path = os.path.join(self.data_dir, file)
            root = os.path.splitext(dat_path)[0]
            samples, rhythm = self.process_prcp(root)
            samples_np = np.array(samples)
            samples_np = np.append(samples_np, rhythm)
            samples = samples_np.tolist()
            self.samples.append(samples)

        self.samples = np.vstack(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.Tensor(self.samples[idx])

    def process_prcp(self, root):
        signals, fields = wfdb.rdsamp(root)
        signal = signals[:, 0]

        new_sig = resample(signal, len(signal) // 5) # Downsample to 100 Hz
        sample = nk.ecg_clean(new_sig, sampling_rate=100)
        samples = StandardScaler().fit_transform(sample.reshape(-1, 1))

        # Extract rhythm information from fields
        rhythm_str = fields['comments'][3]
        if 'Sinus rhythm.' in rhythm_str:
            rhythm = [0]
        else:
            rhythm = [1]

        return samples, rhythm


if __name__ == '__main__':
    path = "/home/kavra/Datasets/medical/LUDB/"
    dataset = LUDBDataset(path)
    print(dataset[0].size())