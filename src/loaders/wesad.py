import os, pickle
import numpy as np
import neurokit2 as nk
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler
from torch.utils import data
from tqdm import tqdm

os.makedirs("data/wesad", exist_ok=True)


class WESAD(data.Dataset):
    def __init__(self, root, sr):
        super().__init__()
        self.root = root
        self.sr = sr
        self.win = sr * 10
        self.pps = list(range(1, 18))
        self.pps = [f"S{i}" for i in self.pps]
        self.pps.remove("S1")
        self.pps.remove("S12")

        ecg_all, lab_all, names = [], [], []
        print("Loading participant data...")
        for p in tqdm(self.pps):

            if os.path.exists(f"data/wesad/{p}_ecg_3.npy"):
                ecg = np.load(f"data/wesad/{p}_ecg.npy")
                ecg = StandardScaler().fit_transform(ecg)
                lab = np.load(f"data/wesad/{p}_lab.npy")
            else:
                with open(f"{self.root}/{p}/{p}.pkl", "rb") as f:
                    data = pickle.load(f, encoding="latin1")

                lab = data["label"]
                ecg = data["signal"]["chest"]["ECG"][:, 0]

                baseline = ecg[lab == 1]
                baseline = self.preprocess(baseline)
                lab1 = np.ones(len(baseline))

                stress = ecg[lab == 2]
                stress = self.preprocess(stress)
                lab2 = np.ones(len(stress)) * 2

                amusement = ecg[lab == 3]
                amusement = self.preprocess(amusement)
                lab3 = np.ones(len(amusement)) * 3

                meditation = ecg[lab == 4]
                meditation = self.preprocess(meditation)
                lab4 = np.ones(len(meditation)) * 4

                ecg = np.concatenate([baseline, stress, amusement])  # , meditation])
                lab = np.concatenate([lab1, lab2, lab3])  # , lab4])

                np.save(f"data/wesad/{p}_ecg_3.npy", ecg)
                np.save(f"data/wesad/{p}_lab_3.npy", lab)

            ecg_all.append(ecg)
            lab_all.append(lab)
            names += [p] * len(ecg)

        self.samples = np.vstack(ecg_all)
        self.labels = np.concatenate(lab_all) - 1
        self.names = names

        print(f"Loaded {len(self.labels)} ECG samples in total.")

    def preprocess(self, ecg):
        # downsample to 100Hz
        new_len = int((len(ecg) / 700) * self.sr)
        proc = resample_poly(ecg, self.sr, 700)[:new_len]
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
    root = "/home/kavra/Datasets/physio/WESAD/"
    dataset = WESAD(root, sr=100)
    print(dataset[0][0].shape, dataset[0][1], dataset[0][2])
