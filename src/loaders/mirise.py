import os, pickle, neurokit2 as nk
import numpy as np, pandas as pd
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler
from torch.utils import data
from tqdm import tqdm

os.makedirs("data/mirise", exist_ok=True)


class MIRISE(data.Dataset):
    def __init__(self, root, sr, cat="str"):
        super().__init__()
        self.root = root
        self.sr = sr
        self.win = sr * 10
        self.cat = cat
        self.pps = list(range(1, 20))
        self.pps.remove(7)
        self.pps.remove(8)
        self.pps = [f"00{i}" if i < 10 else f"0{i}" for i in self.pps]

        ecg_all, eda_all, lab_all, names = [], [], [], []
        print("Loading participant data...")
        for p in tqdm(self.pps):

            if os.path.exists(f"data/mirise/{p}_ecg.pkl"): # and False:
                with open(f"data/mirise/{p}_ecg.pkl", "rb") as f:
                    ecg = pickle.load(f)
                with open(f"data/mirise/{p}_eda.pkl", "rb") as f:
                    eda = pickle.load(f)
                with open(f"data/mirise/{p}_lab_{self.cat}.pkl", "rb") as f:
                    lab = pickle.load(f)
            else:
                ecg, eda = {}, {}
                lab = {cat: {} for cat in ["drw", "ftg", "str"]}
                for condition in os.listdir(f"{self.root}/{p}"):
                    # load ECG data
                    data = pd.read_csv(
                        f"{self.root}/{p}/{condition}/PolymatePhysiologicalSignals.csv"
                    )
                    this_ecg = self.preprocess_ecg(data["ecg"])[:360]
                    this_ecg = StandardScaler().fit_transform(this_ecg)
                    this_eda = self.preprocess_eda(data["gsr"])[:360]
                    this_eda = StandardScaler().fit_transform(this_eda)

                    # load label data
                    label_data = pd.read_csv(
                        f"{self.root}/{p}/{condition}/Annotation.csv"
                    )
                    this_lab = {}
                    for cat in ["drw", "ftg", "str"]:
                        # discard first annotation
                        ldata = [int(i) for i in label_data[cat].values[1:]]
                        this_lab[cat] = []
                        for i in range(len(this_ecg)):
                            # get label for each window
                            temp = ldata[i // (3 * 6)] > 2
                            this_lab[cat].append(int(temp))

                    ecg[condition] = this_ecg
                    eda[condition] = this_eda
                    for cat in ["drw", "ftg", "str"]:
                        lab[cat][condition] = np.array(this_lab[cat])

                with open(f"data/mirise/{p}_ecg.pkl", "wb") as f:
                    pickle.dump(ecg, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"data/mirise/{p}_eda.pkl", "wb") as f:
                    pickle.dump(eda, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                for cat in ["drw", "ftg", "str"]:
                    with open(f"data/mirise/{p}_lab_{cat}.pkl", "wb") as f:
                        pickle.dump(lab[cat], f, protocol=pickle.HIGHEST_PROTOCOL)

            unraveled = [v for _, v in ecg.items()]
            ecg_all.extend(unraveled)
            eda_all.extend([v for _, v in eda.items()])
            lab_all.extend([v for _, v in lab.items()])
            names += [p] * len(np.vstack(unraveled))

        self.samples = {
            "ecg": np.vstack(ecg_all),
            "eda": np.vstack(eda_all),
        }
        self.labels = np.concatenate(lab_all)
        self.names = names

        print(f"Loaded {len(self.labels)} ECG samples in total.")

    def preprocess_ecg(self, ecg):
        # downsample to 100Hz
        new_len = int((len(ecg) / 500) * self.sr)
        proc = resample_poly(ecg, self.sr, 500)[:new_len]
        # BP filtering
        proc = nk.ecg_clean(proc, sampling_rate=self.sr)
        # segment into 10 sec windows
        proc = [proc[i : i + self.win] for i in range(0, len(proc), self.win)]
        # discard first and last window
        return np.stack(proc[1:-1])
    
    def preprocess_eda(self, eda):
        # downsample to 10Hz
        new_len = int((len(eda) / 500) * 10)
        proc = resample_poly(eda, 10, 500)[:new_len]
        # BP filtering
        proc = nk.eda_clean(proc, sampling_rate=10)
        # segment into 10 sec windows
        win = 10 * 10
        proc = [proc[i : i + win] for i in range(0, len(proc), win)]
        # discard first and last window
        return np.stack(proc[1:-1])

    def __len__(self):
        return len(self.samples["ecg"])

    def __getitem__(self, idx):
        d = {
            "ecg": self.samples["ecg"][idx],
            "eda": self.samples["eda"][idx],
        }
        return d, self.labels[idx], self.names[idx]


if __name__ == "__main__":
    root = "/home/kavra/Datasets/physio/toyota_simulator/"
    dataset = MIRISE(root, sr=100)
    print(dataset[0][0]["eda"].shape, dataset[0][1], dataset[0][2])
