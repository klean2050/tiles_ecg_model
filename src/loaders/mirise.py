import os, pickle, neurokit2 as nk
import numpy as np, pandas as pd
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler
from torch.utils import data
from tqdm import tqdm

os.makedirs("data/mirise", exist_ok=True)


class MIRISE(data.Dataset):
    def __init__(self, root, sr):
        super().__init__()
        self.root = root
        self.sr = sr
        self.win = sr * 10
        self.pps = list(range(1, 20))
        self.pps.remove(7)
        self.pps.remove(8)
        self.pps = [f"00{i}" if i < 10 else f"0{i}" for i in self.pps]

        ecg_all, lab_all, names = [], [], []
        print("Loading participant data...")
        for p in tqdm(self.pps):

            if os.path.exists(f"data/mirise/{p}_ecg.pkl"):
                with open(f"data/mirise/{p}_ecg.pkl", "rb") as f:
                    ecg = pickle.load(f)
                with open(f"data/mirise/{p}_lab.pkl", "rb") as f:
                    lab = pickle.load(f)
            else:
                ecg, lab = {}, {}
                for condition in os.listdir(f"{self.root}/{p}"):
                    data = pd.read_csv(
                        f"{self.root}/{p}/{condition}/PolymatePhysiologicalSignals.csv"
                    )

                    this_ecg = self.preprocess(data["ecg"])
                    this_lab = (
                        np.zeros(len(this_ecg))
                        if condition == "sunny"
                        else np.ones(len(this_ecg))
                    )
                    ecg[condition] = this_ecg
                    lab[condition] = this_lab

                with open(f"data/mirise/{p}_ecg.pkl", "wb") as f:
                    pickle.dump(ecg, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"data/mirise/{p}_lab.pkl", "wb") as f:
                    pickle.dump(lab, f, protocol=pickle.HIGHEST_PROTOCOL)

            unraveled = [v for _, v in ecg.items()]
            ecg_all.extend(unraveled)
            lab_all.extend([v for _, v in lab.items()])
            names += [p] * len(np.vstack(unraveled))

        self.samples = np.vstack(ecg_all)
        self.samples = StandardScaler().fit_transform(self.samples)
        self.labels = np.concatenate(lab_all)
        self.names = names

        print(f"Loaded {len(self.labels)} ECG samples in total.")

    def preprocess(self, ecg):
        # downsample to 100Hz
        new_len = int((len(ecg) / 500) * self.sr)
        proc = resample_poly(ecg, self.sr, 500)[:new_len]
        # BP filtering
        proc = nk.ecg_clean(proc, sampling_rate=self.sr)
        # segment into 10 sec windows
        proc = [proc[i : i + self.win] for i in range(0, len(proc), self.win)]
        # discard first and last window
        return np.stack(proc[1:-1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ecg = self.samples[idx]
        lab = self.labels[idx]
        nam = self.names[idx]
        return ecg, lab, nam


if __name__ == "__main__":
    root = "/home/kavra/Datasets/toyota_simulator/"
    dataset = MIRISE(root, sr=100)
    print(dataset[0][0].shape, dataset[0][1], dataset[0][2])
