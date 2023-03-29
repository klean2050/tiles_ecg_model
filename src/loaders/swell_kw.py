import glob, os, tqdm
from torch.utils import data
from scipy.signal import resample_poly
from collections import defaultdict
import numpy as np, pandas as pd


class SWELL_KW(data.Dataset):
    def __init__(self, root, sr, gtruth=3):
        super().__init__()
        self.root = root
        self.gtruth = gtruth
        self.physio_path = self.root + "0 - Raw data/D - Physiology - raw data/"
        self.signal_path = self.physio_path + "csv_extracted_signals/"
        self.label_path = self.root + "Behavioral-features - per minute.xlsx"
        self.cache_path = "data/swell_kw/"
        self.window_size = sr * 10

        if os.path.exists(self.cache_path):
            print("Loading cached dataset ...")
            self.samples = np.load(self.cache_path + "samples.npy")
            self.labels = np.load(self.cache_path + "labels.npy")
            self.names = np.load(self.cache_path + "names.npy")
        else:
            os.makedirs(self.cache_path, exist_ok=True)

            print("Loading label data ...")
            label = pd.ExcelFile(self.label_path, engine="openpyxl")
            label_sheet_names = label.sheet_names
            labels = label.parse(label_sheet_names[0])  # we only need sheet 1
            swell_labels = labels.drop_duplicates(subset=["PP", "Blok"], keep="last")
            swell_labels = swell_labels.reset_index(drop=True)

            self.samples, self.labels, self.names = [], [], []

            print("\nLoading participant data ...")
            for pp, files in self.iterate_subjects():

                # load ECG data of participant <pp>
                pp_ecg = []
                for file in files:
                    with open(file, "r") as f:
                        signal = f.readlines()[0].split(",")
                        signal = [float(s) for s in signal]

                    new_len = int((len(signal) / 2048) * 100)
                    downsampled = resample_poly(signal, 100, 2048)[:new_len]
                    pp_ecg.append(downsampled)

                pp_mean = np.mean(np.concatenate(pp_ecg), axis=0)
                pp_std = np.std(np.concatenate(pp_ecg), axis=0)

                for i, signal in enumerate(pp_ecg):
                    signal -= pp_mean
                    signal /= pp_std
                    these_windows = self.make_windows(signal)
                    self.samples += [these_windows]
                    self.names += [pp] * len(these_windows)

                    label_set = swell_labels[
                        (swell_labels["PP"] == pp) & (swell_labels["Blok"] == i + 1)
                    ]
                    label_set = np.asarray(
                        label_set[
                            [
                                "Valence_rc",
                                "Arousal_rc",
                                "Dominance",
                                "Stress",
                                "Frustration",
                            ]
                        ]
                    )

                    if not len(label_set):
                        these_wlabels = [self.labels[-1]] * len(these_windows)
                    else:
                        these_wlabels = [label_set[0]] * len(these_windows)
                    self.labels += these_wlabels

            self.samples = np.vstack(self.samples)
            self.labels = np.vstack(self.labels)

            # save to cache
            np.save(self.cache_path + "samples.npy", self.samples)
            np.save(self.cache_path + "labels.npy", self.labels)
            np.save(self.cache_path + "names.npy", self.names)

        print(f"Loaded {len(self.labels)} ECG samples in total.")

    def iterate_subjects(self):
        files = glob.glob(self.signal_path + "*.csv")
        person_to_files = defaultdict(list)
        fnames = [os.path.basename(f) for f in files]
        for i, f in zip(fnames, files):
            p_name = i[: i.find("_")].upper()
            person_to_files[p_name].append(f)
        for p in tqdm.tqdm(person_to_files.keys()):
            yield p, person_to_files[p]

    def make_windows(self, ecg_array):
        ecg_array = [
            ecg_array[i : i + self.window_size]
            for i in range(0, len(ecg_array), self.window_size)
        ]
        # discard first and last window (distorted)
        return np.stack(ecg_array[1:-1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ecg = self.samples[index]
        label = self.labels[index]
        label = label[self.gtruth] > 4.5
        name = self.names[index]
        return ecg, label, name


if __name__ == "__main__":
    root = "/home/kavra/Datasets/physio/SWELL_KW/"
    dataset = SWELL_KW(root, sr=100)
    print(dataset[0][0].shape, dataset[0][1])
